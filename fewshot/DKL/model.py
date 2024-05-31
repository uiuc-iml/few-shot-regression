import os
import torch
import torch.nn as nn
import gpytorch
from ..model import AdaptiveModel, ProbabilisticModel, SupervisedLearningProblem, TorchModelModule
from ..GP.model import GP
from ..utils import TorchTrainableModel, totorch
from ..data import FlatDataset
from .. import utils
from dataclasses import replace

class DKL(TorchModelModule):
    """
    Deep Kernel Learning model implementation.

    If few-shot training is enabled (i.e., k_shot > 0), the model is equivalent
    to Deep Kernel Transfer (Pattachiola et al 2020).

    Predicts backbone -> feature_model -> GP_models -> output prediction

    Parameters:
        - feature_hidden_layers (list): list of hidden layer sizes for the feature model
        - GP_input_dim (int): the input dimension of the GP model
        - GP_model_type (str): the type of GP model to use, either 'exact' or 'approximate'
    """
    def __init__(self, problem : SupervisedLearningProblem, params : dict):
        super(DKL, self).__init__(problem)

        self.params = params

        x_dims = self.problem.core_input_shape()[0]
        hidden_size = [x_dims]+params.get('feature_hidden_layers',[128])+[params['GP_input_dim']]

        self.MLP = utils.MLP(hidden_size)
        inner_problem = replace(problem, input_shape=(params['GP_input_dim'],), input_vars=None, x_encoder=None, encoded_input_shape=None)
        gp_params = params.copy()
        #disable scaling per dimension
        gp_params['kernel_independent_scaling'] = False
        self.GP = GP(inner_problem, gp_params)

    @torch.no_grad()
    def set_train_data(self, support_x, support_y):
        if len(support_x)==0:
            self.GP.set_train_data(support_x,support_y)
            return

        GP_inputs = self.MLP(support_x)
        self.GP.set_train_data(GP_inputs, support_y)

    @torch.no_grad()
    def condition_core(self, support_x, support_y):
        if len(support_x)==0:
            self.GP.condition_core(support_x,support_y)
            return

        GP_inputs = self.MLP(support_x)
        self.GP.condition_core(GP_inputs, support_y)

    def forward_core(self, x):
        GP_inputs = self.MLP(x)
        return self.GP.forward_core(GP_inputs)

    def forward_core_loss(self, x, y) -> tuple:
        GP_inputs = self.MLP(x)
        return self.GP.forward_core_loss(GP_inputs, y)
    
    def forward_core_fewshot_loss(self, support_set, query_set) -> tuple:
        support_x, support_y = support_set
        query_x, query_y = query_set
        support_len = support_x.size(0)

        features = torch.cat((support_x,query_x),axis = -2)
        features = features.view(-1,features.size(-1))
        outputs = torch.cat((support_y,query_y),axis = -2)

        features = self.MLP(features)
        self.GP.set_train_data(features,outputs)
        loss, preds = self.GP.forward_core_loss(features,outputs)
        return loss, preds[support_len:]



class DKLModel(TorchTrainableModel,AdaptiveModel,ProbabilisticModel):
    def __init__(self, problem:SupervisedLearningProblem, params=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
        TorchTrainableModel.__init__(self,problem,params)
        self.model = DKL(problem,params).to(self.device)
        
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

    def train_step(self,x,y) -> tuple:
        if self.fewshot_training():
            support_x,support_y = x
            query_x,query_y = y
            return self.model.forward_fewshot_loss(x,y)  #will handle setting the training data
        else:
            self.model.set_train_data(x,y)
            return self.model.forward_loss(x,y)
    
    def test_step(self,x,y) -> tuple:
        if self.fewshot_training():
            support_x,support_y = x
            query_x,query_y = y
            return self.model.forward_fewshot_loss(x,y)  #will handle setting the training data
        else:
            self.model.set_train_data(x,y)
            return self.model.forward_loss(x,y)
    
    def __call__(self, x):
        """Override ProbabilisticModel"""
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(False):
            return super(DKLModel,self).__call__(x)