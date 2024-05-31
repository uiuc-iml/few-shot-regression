import os
from typing import Union,Optional,List
import gpytorch
#from pytest import param
import torch
import torch.nn as nn

from ..GP.model import GP
from ..model import TrainableModel,ProbabilisticModel,AdaptiveModel,SupervisedLearningProblem,TorchModelModule
from ..utils import TorchTrainableModel,weights_init, totorch
from .. import utils
from dataclasses import replace


class Encoder(nn.Module):
    
    def __init__(self, sizes):
        super(Encoder, self).__init__()
        self._sizes = sizes
        self.linears = utils.MLP(sizes, batchnorm=False)

    def forward(self, support_x, support_y):
        """
        Encode training elements and aggregate the resulting representations
        into a single vector representation
        Args:
            support_x:  (task_batch_size) x context_set_size x feature_dim
            support_y:  (task_batch_size) x context_set_size x outcome_dim
        Returns:
            representations: context_set_size x representation_dim 
        """
        x = torch.cat((support_x, support_y), dim=-1)
        x = self.linears(x)
        out_shape = list(x.shape)
        out_shape[-1] = -1
        representations = x.view(out_shape)
        return representations

    def init_weights(self):
        self.apply(weights_init)

class Aggregator(nn.Module):
    """Performs an aggregation of the inputs.  Can include an MLP to process the output."""

    def __init__(self, variance = True, mlp_sizes : Optional[List[int]] = None):
        super(Aggregator, self).__init__()
        self.variance = variance
        if mlp_sizes is not None:
            self.linears = utils.MLP(mlp_sizes, batchnorm=False)
        else:
            self.linears = nn.Identity()

    def forward(self, representations):
        """
        Aggregate the representations produced by the encoder in an order-invariant manner
        with empirical mean (first moment) and standard deviatiation (second moment). 
        Create a nonlinear mapping of the first 
        Args:
            representations: (task_batch) x (context_set_size) x representation_dim
        Returns:
            aggregated_representation: (task_batch) x output_dim where output_dim = 
            representation_dim if variance is False, 2*representation_dim if variance
            is True, and mlp_sizes[-1] if mlp_sizes is not None
        """
        representation_dim = representations.shape[-1]
        out_shape = list(representations.shape)
        out_shape[-1] = representation_dim
        if self.variance:
            if representations.shape[-2]==1:
                mean = representations
                var = torch.zeros_like(mean)
            else:
                mean = representations.mean(dim=-2)
                var = representations.var(dim=-2)
            x = torch.cat((mean, var), dim=-1)
        else:
            x = representations.mean(dim=-2)
        return self.linears(x)
    
    def init_weights(self):
        self.apply(weights_init)

class Decoder(nn.Module):

    def __init__(self, sizes):
        super(Decoder, self).__init__()
        self.linears = utils.MLP(sizes, batchnorm=False)

    def forward(self, query_x, representation):
        """
        General module for input-target pair representation
        Take representation of current context and optionally 
        query to compute the kernel and mean of the GP.
        Args:
            representation: (task_batch) x representation_dim
            query_x: (task_batch) x (query_set_size) x feature_dim
        Returns:
            out: mean or kernel parameter size depending on the module
            (task_batch) x (query_set_size) x out_feature_dim
        """
        out_shape = list(query_x.shape)[:-1]+[-1]
        representation_dim = representation.shape[-1]
        input_dim = query_x.shape[-2]
        feature_dim = query_x.shape[-1]

        query_x = query_x.view(-1, feature_dim)
        representation = representation.unsqueeze(-2).repeat(1, input_dim, 1)
        representation = representation.view(-1, representation_dim)

        input = torch.cat((representation, query_x), dim=-1)
        x = input.view(out_shape)
        x = self.linears(x)
        out = x.view(out_shape)
        return out
    
    def init_weights(self):
        self.apply(weights_init)


class ADKLCore(nn.Module):
    """ADKL Core model for few-shot learning
    
    Uses a deep kernel learning approach but with a task encoder.  The input
    to the GP is the pair (decoder(input_encoder(x[i]),z[1...k]), y[i]) where
    
    z[1,...,k] = aggregator(encoder(input_encoder(x[j]),output_encoder(y[j])) for j in 1...k))
    
    is the task encoding.
    """
    def __init__(self, x_dims, y_dims, params):
        super(ADKLCore, self).__init__()
        self.params = params

        # Loading Deep Kernel architecture
        if 'u_hlayers' in params:
            u_sizes = [x_dims] + params['u_hlayers'] + [params['d_x']]
            enc_x_len = params['d_x']
            self.input_encoder = utils.MLP(u_sizes, batchnorm=False)
        else:
            self.input_encoder = nn.Identity()
            enc_x_len = x_dims
        
        if 'v_hlayers' in params:
            v_sizes = [y_dims] + params['v_hlayers'] + [params['d_y']]
            self.output_encoder = utils.MLP(v_sizes, batchnorm=False)
            enc_y_len = params['d_y']
        else:
            self.output_encoder = nn.Identity()
            enc_y_len = y_dims

        encoder_sizes = [enc_x_len + enc_y_len] + params['encoder_hlayers'] + [params['task_representation_dim']]
        self.encoder = Encoder(encoder_sizes) #r

        if 'aggregator_hlayers' in params:
            #uses an MLP to process (mean,variance) of representations
            aggregator_variance = True
            aggregator_sizes = [encoder_sizes[-1]*2] + params['aggregator_hlayers'] + [params['task_representation_dim']]
        else:
            #just computes the mean of the representations
            aggregator_variance = False
            aggregator_sizes = None
        self.aggregator = Aggregator(variance=aggregator_variance,mlp_sizes=aggregator_sizes)#w
        
        #TODO: include an option for a decoder that just concatenates the task and input
        decoder_sizes = [aggregator_sizes[-1] + enc_x_len] + params['decoder_hlayers'] + [params['GP_input_dim']]
        self.decoder = Decoder(decoder_sizes)#o

        gp_problem = SupervisedLearningProblem(input_shape=(params['GP_input_dim'],),output_shape=(enc_y_len,))
        gp_params = params.copy()
        gp_params['kernel_independent_scaling'] = False
        self.GP = GP(gp_problem, gp_params)
        self.phi = None  #current encoding of support date

    def compute_task_encoder_loss(self, phis_train, phis_test):
        if len(phis_train.shape) == 1:
            phis_train = phis_train.unsqueeze(0)
        if len(phis_test.shape) == 1:
            phis_test = phis_test.unsqueeze(0)
        
        def compute_cos_similarity(query, support):
            norms_query = torch.norm(query, p=None, dim=-1)
            norms_support = torch.norm(support, p=None, dim=-1)
            num = torch.mm(query, support.t())
            deno = norms_query.unsqueeze(1) * norms_support.unsqueeze(0) + 1e-10
            return num / deno

        # train x test
        set_code = compute_cos_similarity(phis_train, phis_test)

        y_preds_class = torch.arange(len(set_code), device=set_code.device)

        accuracy = (set_code.argmax(dim=1) == y_preds_class).sum().item() / len(set_code)
        #print('Accuracy ' + str(accuracy))

        b = set_code.size(0)
        mi = set_code.diagonal().mean() 
        if b > 1:
             #encourage differences between task encodings amongst the batch
             mi -= torch.log((set_code * (1 - torch.eye(b, device=set_code.device))).exp().sum() / (b * (b - 1)))
        loss = - mi

        return loss, accuracy

    def deep_map_support(self, support_x, support_y):
        support_x = self.input_encoder(support_x)
        support_y = self.output_encoder(support_y)
        support_features = self.encoder(support_x, support_y)
        support_representation = self.aggregator(support_features)
        gp_x = self.decoder(support_x, support_representation)
        return support_representation, gp_x

    def forward(self, x):
        assert self.phi is not None, "Support set was not provided"
        x = self.input_encoder(x)
        query_features = self.decoder(x, self.phi)
        mean,std = self.GP(query_features)
        return mean,std

    def set_train_data(self, support_x, support_y):
        print("Setting training data of size",support_x.size(),support_y.size())
        if len(support_x) == 0:
            self.phi = torch.zeros(self.params['task_representation_dim'], device=support_x.device)
            self.GP.set_train_data(torch.Tensor([]).to(support_x.device), torch.Tensor([]).to(support_x.device))
            return
        phi_train, support_features = self.deep_map_support(support_x, support_y)
        self.phi = phi_train
        self.GP.set_train_data(support_features, support_y)


class ADKLModule(TorchModelModule):
    def __init__(self, problem : SupervisedLearningProblem, params : dict):
        TorchModelModule.__init__(self, problem)
        x_dims = problem.core_input_shape()[0]
        y_dims = problem.core_output_shape()[0]

        # Create ADKL_Model
        self.model = ADKLCore(x_dims, y_dims, params)

    @torch.no_grad()
    def condition_core(self, support_x, support_y):
        self.model.set_train_data(support_x, support_y)

    def forward_core(self, x : torch.Tensor):
        return self.model(x)

    def forward_core_fewshot_loss(self, support_data, query_data):
        support_x, support_y = support_data
        query_x, query_y = query_data

        # Deep Kernel
        self.train()
        phi_train, support_features = self.model.deep_map_support(support_x, support_y)
        phi_test, query_features = self.model.deep_map_support(query_x, query_y)
        self.model.GP.set_train_data(support_features, support_y)
        
        # Gaussian Process -- do we want to train the kernel?
        self.eval()
        loss, preds = self.model.GP.forward_core_fewshot_loss((support_features,support_y),(query_features,query_y))

        #Add task encoder loss * a weight. 
        task_encoder_loss, task_accuracy = self.model.compute_task_encoder_loss(phi_train, phi_test)
        loss += self.model.params['task_encoder_loss_weight']*task_encoder_loss
        
        return loss, preds
    


class ADKLModel(TorchTrainableModel,ProbabilisticModel,AdaptiveModel):
    """Adaptive DKL (Toussou et al 2020)

    Combines a task encoding like CNP with a Deep Kernel Learning approach.

    Configurable parameters:
    - u_hlayers (optional): list of hidden layer sizes for (core) input encoder u.  If not present, the identity is used
    - v_hlayers (optional): list of hidden layer sizes for (core) output encoder v.  If not present, the identity is used
    - encoder_hlayers: list of hidden layer sizes for (x,y) encoder
    - aggregator_hlayers (optional): list of hidden layer sizes for aggregator.  If not present, only a mean is computed.
    - decoder_hlayers: list of hidden layer sizes for decoder
    - task_representation_dim: dimension of task representation (output of encoder/aggregator)
    - GP_input_dim: input dimension of GP corresponding to input x + task representation (output of decoder)
    - task_encoder_loss_weight: weight of task encoder loss in the total loss
    - kernel: kernel function for the GP
    - kernel_num_mixtures: number of mixtures for the Spectral Mixture Kernel
    - loss: loss function for the GP (marginal or conditional)

    """
    def __init__(self, problem : SupervisedLearningProblem, params=None, benchmark=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
            params = utils.load_config(params)
        super().__init__(problem,params)

        self.model = ADKLModule(problem,params).to(self.device)
        # Clear support set
        self.reset()
    
    def reset(self):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.Tensor([]).float().to(self.device)
        self.support_y = torch.Tensor([]).float().to(self.device)
        self.model.condition_core(self.support_x,self.support_y)
        
    def update(self,x,y):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.cat((self.support_x,self.problem.encode_input(totorch(x).unsqueeze(0).float().to(self.device))))
        self.support_y = torch.cat((self.support_y,self.problem.encode_output(totorch(y).unsqueeze(0).float().to(self.device))))
        self.model.condition_core(self.support_x,self.support_y)

    def __call__(self, x):
        """Override ProbabilisticModel"""
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
            super(ADKLModel,self).__call__(x)
