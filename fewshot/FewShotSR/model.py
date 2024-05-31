import torch
import torch.nn as nn
import os, glob
import gpytorch
from .. import utils
from ..model import AdaptiveModel,ProbabilisticModel,TorchModelModule,SupervisedLearningProblem
from ..ADKL.model import Encoder, Aggregator, Decoder
from ..GP.model import GP, _ExactGPModule
from ..utils import MLP
from linear_operator import LinearOperator
from ..utils import TorchTrainableModel, totorch

class GPWithTaskNoiseModel(gpytorch.models.ExactGP):
    """Implements equation (2) in the paper"""
    def __init__(self, base_model : _ExactGPModule, index:int, y_dims:int):
        super(GPWithTaskNoiseModel, self).__init__(base_model.train_inputs, base_model.train_targets, base_model.likelihood)
        self.base_model = base_model
        self.index = index
        self.y_dims = y_dims
        assert index >=0 and index < y_dims

    def forward(self, xfb):
        x = xfb[:, :self.y_dims]
        fb = xfb[:, -self.y_dims+self.index]
        assert torch.all(fb >= 0.0), "fb must be non-negative"
        distribution = self.base_model(x) # type: gpytorch.distributions.MultivariateNormal
        if isinstance(distribution._covar, LinearOperator):
            distribution._covar = distribution._covar.add_diagonal(fb)
        else:
            distribution._covar = torch.diagonal_scatter(distribution._covar, torch.diagonal(distribution._covar,dim1=-2,dim2=-1)+fb, dim1=-2,dim2=-1)
        return distribution

class FewShotSRCore(nn.Module):
    """
    Refer to Fig. 2 of https://arxiv.org/pdf/2010.04360.pdf
    """
    def __init__(self, x_dims, y_dims, params):
        super(FewShotSRCore, self).__init__()

        self.params = params

        z_dims = params['task_representation_dim']
        encoder_sizes = [x_dims + y_dims] + params['encoder_hlayers'] + [z_dims]
        if 'aggregator_hlayers' in params:
            aggregator_sizes = [encoder_sizes[-1]] + params['aggregator_hlayers'] + [z_dims]
        else:
            aggregator_sizes = None
        f_b_sizes = [z_dims] + self.params['f_b_hlayers'] + [1]
        f_k_sizes = [z_dims  + x_dims] + self.params['f_k_hlayers'] + [params['d_fk_out']]
        f_m_sizes = [z_dims  + x_dims] + self.params['mean_hlayers'] + [y_dims]

        self.encoder = Encoder(encoder_sizes)
        self.aggregator = Aggregator(False,aggregator_sizes)
        self.f_b = MLP(f_b_sizes)
        self.f_k = Decoder(f_k_sizes)
        self.f_m = Decoder(f_m_sizes)

        gp_problem = SupervisedLearningProblem(input_shape=(f_k_sizes[-1],),output_shape=(y_dims,))
        gp_params = params.copy()
        gp_params['kernel_independent_scaling'] = False
        gp_params['multi_output'] = False
        gp_params['zero_mean'] = True
        self.GP = GP(gp_problem,gp_params)
        assert y_dims == len(self.GP.gmodel.channel_models), "y_dims must be equal to the number of tasks"
        for i in range(y_dims):
            self.GP.gmodel.channel_models[i] = GPWithTaskNoiseModel(self.GP.gmodel.channel_models[i],i,y_dims)
        
        #conditioning
        self.task_representation = None
        self.f_b_output = None
    
    def set_train_data(self, context_x, context_y):
        """Set the context set for the task."""
        if len(context_x) < 2 or context_x.shape[0]==0:
            self.task_representation = torch.zeros(self.params['task_representation_dim']).to(context_x.device)
        else:           
            if len(context_x.shape) != 2:
                raise ValueError("context_x must be 2D tensor k x n")
            task_representations = self.encoder(context_x, context_y)
            self.task_representation = self.aggregator(task_representations)
        self.f_b_output = self.f_b(self.task_representation) # used for 2nd term in Eq. 2 in the paper
        
        f_k_output = self.f_k(context_x, self.task_representation) # used for 1st term for Eq. 2 in the paper
        mean_output = self.f_m(context_x, self.task_representation) # used for Eq. 1 in the paper
        
        fk_fb = torch.cat((f_k_output, self.f_b_output.repeat(context_x.shape[0],1)), dim=-1)
        self.GP.set_train_data(fk_fb, context_y - mean_output)

    def forward(self, query_x):
        """
        Takes the task representation of the support set and predict the parameters of the GP.

        Returns:
            mean, std: mean and standard deviation of the prediction
        """
        if self.f_b_output is None:
            raise ValueError("Model not conditioned yet")
        f_k_output = self.f_k(query_x, self.task_representation) # used for 1st term for Eq. 2 in the paper
        mean_output = self.f_m(query_x, self.task_representation) # used for Eq. 1 in the paper
        
        fk_fb = torch.cat((f_k_output, self.f_b_output.repeat(query_x.shape[0])), dim=-1)
        gp_pred, gp_var = self.GP(fk_fb)
        return mean_output + gp_pred, gp_var


    def forward_loss(self, query_x, query_y):
        """
        Takes the task representation of the support set and predict the parameters of the GP.

        Returns:
            loss, preds: loss and mean predictions
        """
        if self.f_b_output is None:
            raise ValueError("Model not conditioned yet")
        f_k_output = self.f_k(query_x, self.task_representation) # used for 1st term for Eq. 2 in the paper
        mean_output = self.f_m(query_x, self.task_representation) # used for Eq. 1 in the paper
        
        fk_fb = torch.cat((f_k_output, self.f_b_output.repeat(query_x.shape[0],1)), dim=-1)
        loss, gp_pred = self.GP.forward_loss(fk_fb, query_y-mean_output)
        return loss, gp_pred+mean_output


class FewShotSR(TorchModelModule):
    def __init__(self, problem : SupervisedLearningProblem, params : dict):
        super(FewShotSR, self).__init__(problem)
        self.params = params

        x_dims = problem.core_input_shape()[0]
        y_dims = problem.core_output_shape()[0]
        self.model = FewShotSRCore(x_dims, y_dims, self.params)

    @torch.no_grad()
    def condition_core(self, support_x, support_y):
        self.model.set_train_data(support_x,support_y)

    def forward_core(self, x):
        return self.model(x)

    def forward_core_fewshot_loss(self, support_data, query_data):
        support_x, support_y = support_data
        query_x, query_y = query_data

        #support_xy = ((support_x, support_out), support_reward)
        self.condition_core(support_x, support_y)
        return self.model.forward_loss(query_x, query_y)


class FewShotSRModel(TorchTrainableModel,AdaptiveModel,ProbabilisticModel):
    def __init__(self,task,params=None,benchmark=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
        assert 'k_shot' in params, "k_shot must be specified in the config file since only few-shot learning is supported."
        TorchTrainableModel.__init__(self,task,params)
        self.model = FewShotSR(task,self.params).to(self.device)
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

