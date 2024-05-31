import os
from typing import Union,Optional
import torch
import torch.nn as nn
import copy

from .. import utils
from ..utils import MLP, TorchTrainableModel
from ..model import AdaptiveModel, SupervisedLearningProblem, TorchModelModule



class DNNFineTuning(TorchModelModule):
    def __init__(self, problem : SupervisedLearningProblem, params : dict):
        super(DNNFineTuning, self).__init__(problem)
        self.problem = problem
        n_inputs = problem.core_input_shape()[0]
        n_outputs = problem.core_output_shape()[0]
        hidden_layers = params.get('hidden_layers',[256])
        hidden_batchnorm = params.get('hidden_batchnorm',False)

        self.model = MLP([n_inputs] + hidden_layers + [n_outputs], batchnorm=hidden_batchnorm)
        self.params = params

        self.loss_fn = nn.MSELoss()
        
    def forward_core(self,x):
        return self.model(x)

    def forward_core_loss(self, x, y):
        y_pred = self.forward_core(x)
        return self.loss_fn(y_pred, y),y_pred


class DNNFineTuningModel(AdaptiveModel,TorchTrainableModel):
    """Implementation of fine tuning.  Backbone is followed by a standard
    fully-connected network.

    Note: not a probabilistic model.

    Configurable parameters (in addition to TorchTrainableModel):
    
        hidden_layers (list of int): # of hidden layers in fully-connected
            network
        hidden_batchnorm (bool): whether to batch-normalize in hidden layers.
        online_lr (float): learning rate
        online_optimizer (str): only supports Adam for now.
        online_train_epochs (int): # of epochs to train for each new support
            data
        
    """
    def __init__(self, problem: SupervisedLearningProblem, params : dict=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
            params = utils.config(params)
        TorchTrainableModel.__init__(self,problem,params)

        self.model = DNNFineTuning(problem,self.params).to(self.device)
        self.original_state_dict = None

    def set_baseline(self):
        """Marks that the current model has been trained and should be used
        for further online training (via update())."""
        self.original_state_dict = copy.deepcopy(self.model.state_dict())

    def reset(self):
        """Overrides AdaptiveScoringModel"""
        if self.original_state_dict is None:
            self.set_baseline()
        else:
            self.model.load_state_dict(self.original_state_dict)
        self.xy_supp = []
        
    def update(self,x,y):
        """Overrides AdaptiveScoringModel.  Update performs fine-tuning on
        all the support data."""
        if self.original_state_dict is None:
            self.set_baseline()
        self.xy_supp.append((x,y))
        
        if len(self.x_supp) > 0:
            for g in self.optimizer.param_groups:
                g['lr'] = self.params['online_lr']
            num_set_params = 0
            for index,param in enumerate(self.model.parameters()):
                num_set_params = index
            for index,param in enumerate(self.model.parameters()):    
                # ic(param,param.requires_grad)
                if num_set_params - index < 2:  #TODO: what is this doing?
                    param.requires_grad = True
                else:
                    param.requires_grad  = False

            self.model.train()
            num_epochs = self.params.get('online_train_epochs',1)
            batchsize = self.params.get('online_train_batch_size',1)
            dataloader = utils.DataLoader(self.xy_supp,batch_size = batchsize, shuffle = True)
            for epoch in range(num_epochs):
                for (self.x_supp,self.y_supp) in dataloader:
                    self.optimizer.zero_grad()
                    self.x_supp = utils.totorch(self.x_supp)
                    self.y_supp = utils.totorch(self.y_supp)
                    loss, p = self.model.forward_loss(self.x_supp,self.y_supp)
                    #print(loss.item())
                    loss.backward()
                    self.optimizer.step()

            self.model.eval()

    def train(self, train_data, val_data = None, writer = None):
        self.original_state_dict = None
        TorchTrainableModel.train(self,train_data,val_data,writer)
        self.set_baseline()
        