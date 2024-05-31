import os
from typing import Union,Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .. import utils
from ..utils import MLP, TorchTrainableModel, totorch, fromtorch
from ..model import AdaptiveModel, SupervisedLearningProblem, TorchModelModule


'''
Functional definitions of common layers
Useful for when weights are exposed rather 
than being contained in modules
'''

def linear(input, weight, bias=None):
    if bias is None:
        return F.linear(input, weight.cuda())
    else:
        return F.linear(input, weight.cuda(), bias.cuda())

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight.cuda(), bias.cuda(), stride, padding, dilation, groups)

def relu(input):
    return F.threshold(input, 0, 0, inplace=True)

def leaky_relu(input, negative_slope=0.0):
    return F.leaky_relu(input, negative_slope)

def sigmoid(input):
    return torch.sigmoid(input)

def maxpool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)

def avgpool(input, kernel_size, stride=None):
    return F.avg_pool2d(input, kernel_size, stride)

def dropout(input, p=0.0):
    return F.dropout(input, p=p)

def local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    return F.local_response_norm(input, size, alpha, beta, k)

def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch '''
    # This hack only works when momentum is 1 and avoids needing to track running stats
    # by substuting dummy variables
    running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).cuda()
    running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).cuda()
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def bilinear_upsample(in_, factor):
    return F.upsample(in_, None, factor, 'bilinear')

def log_softmax(input):
    return F.log_softmax(input)

class MAML(TorchModelModule):
    """Parameters:
        - hidden_layers (list): list of hidden layer sizes
        - hidden_batchnorm (bool): whether to use batchnorm in hidden layers
        - dropout_p (float): dropout probability
        - num_updates (int): number of updates during meta training
        - update_lr (float): learning rate for updates

    """
    def __init__(self, problem : SupervisedLearningProblem, params):
        super(MAML, self).__init__(problem)
        self.params = params
        
        hidden_layers = params.get('hidden_layers',[4])
        hidden_batchnorm = params.get('hidden_batchnorm', False) # currently works only with False
        hidden_dropout = params.get('dropout_p', 0.0)

        input_dims = problem.core_input_shape()[0]
        output_dims = problem.core_output_shape()[0]
        self.fc = MLP([input_dims] + hidden_layers + [output_dims],
                          batchnorm=hidden_batchnorm, output='Sigmoid', dropout=hidden_dropout)

        self.loss_fn = nn.MSELoss() # MSELoss works well for grasping task
        
        self.num_updates = params['num_updates']
        self.update_lr = params['update_lr']

    def forward(self, x, weights=None):
        if weights is not None:
            oldweights = self.state_dict()
            self.load_state_dict(weights)
            y = super(MAML,self).forward(x)
            self.load_state_dict(oldweights)
            return y
        return super(MAML,self).forward(x)

    def forward_core(self, x):   
        return self.fc(x)

    def condition_core(self, support_x, support_y):
        # query_x = self.task.encode_observation_action(query_obs,query_act)
        bz = support_y.shape[0]

        self.fc.train()
        optimizer = torch.optim.SGD(self.fc.parameters(), lr=self.update_lr)

        # mimic the parameter updates from the training loop
        for batch in range(bz):
            s_i, s_l = support_x[batch], support_y[batch]
            for _ in range(self.num_updates):
                s_out = self.forward(s_i, s_l)
                s_loss = self.loss_fn(s_out, s_l)
                optimizer.zero_grad()
                s_loss.backward()
                optimizer.step()
        self.fc.eval()



class MAMLModel(TorchTrainableModel,AdaptiveModel):
    def __init__(self,problem : SupervisedLearningProblem, params):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
        assert 'k_shot' in params, "k_shot must be specified in the config file since only few-shot learning is supported."
        TorchTrainableModel.__init__(self,problem,params)
        self.model = MAML(problem,self.params)
        self.reset()

    def train_step(self, support_data, query_data):
        support_x, support_y = support_data
        query_x, query_y = query_data

        bz = support_y.shape[0]

        query_outputs = []
        query_loss = 0.0

        # updated weights during meta training, based on https://github.com/katerakelly/pytorch-maml/tree/master
        fast_weights = self.model.state_dict()
        meta_grads = []
        for batch in range(bz):
            s_i, s_l = support_x[batch], support_y[batch]
            for n_update in range(self.model.num_updates):
                if n_update == 0:
                    s_out = self.model.forward(s_i)
                    s_loss = self.model.loss_fn(s_out, s_l)
                    for p in self.model.parameters(): p.grad = None
                    grads = torch.autograd.grad(s_loss, self.model.parameters(), create_graph=False)
                else:
                    s_out = self.model.forward(s_i, fast_weights)
                    s_loss = self.model.loss_fn(s_out, s_l)
                    for p in fast_weights.values(): p.grad = None
                    grads = torch.autograd.grad(s_loss, fast_weights.values(), create_graph=False)
            fast_weights = OrderedDict((name, param - self.model.update_lr*grad) for ((name, param), grad) in zip(fast_weights.items(), grads))

            # compute gradients for meta optimization w.r.t query set
            q_i, q_l = query_x[batch], query_y[batch]
            q_out = self.model.forward(q_i, fast_weights)
            q_loss = self.model.loss_fn(q_out, q_l)
            for p in self.model.parameters():
                p.grad = None
            grads = torch.autograd.grad(q_loss, self.model.parameters(), create_graph=False, allow_unused=True)
            m_grads = {name:g for ((name, _), g) in zip(self.model.named_parameters(), grads)}
            meta_grads.append(m_grads)

            # accumulate query outputs
            query_outputs.append(q_out)
            query_loss += q_loss

        # accumulate gradients from earlier
        gradients = {}
        for k in meta_grads[0].keys():
            sum_grad = 0.
            for d in meta_grads:
                if d[k] is not None: # if a module in network is not used then its gradient is None
                    sum_grad += d[k]
                else:
                    sum_grad += torch.zeros_like(fast_weights[k])
            gradients[k] = sum_grad

        # meta optimization
        self.optimizer.zero_grad()
        for (k, v) in self.model.named_parameters():
            v.grad = gradients[k] # assign the computed gradients to model parameters
        self.optimizer.step() # meta update with meta gradients computed above

        query_outputs = torch.stack(query_outputs, dim=0)
        return query_loss, query_outputs

    def reset(self):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.Tensor([]).float().to(self.device)
        self.support_y = torch.Tensor([]).float().to(self.device)
        self.model.condition_core(self.support_x,self.support_y)
        
    def update(self,x,y):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.cat((self.support_x,self.problem.encode_input(totorch(x).float().to(self.device))))
        self.support_y = torch.cat((self.support_y,self.problem.encode_output(totorch(y).float().to(self.device))))
        self.model.condition_core(self.support_x,self.support_y)
