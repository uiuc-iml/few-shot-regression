import os
import torch
import gpytorch
import gpytorch.constraints
from torch.utils.tensorboard import SummaryWriter
from linear_operator import to_linear_operator
from linear_operator.operators import KroneckerProductLinearOperator
from ..model import AdaptiveModel, ProbabilisticModel, TorchModelModule, SupervisedLearningProblem
from ..utils import TorchTrainableModel, totorch
from ..data import MultiTaskDataset
import copy
from dataclasses import dataclass, asdict, field
from sklearn.decomposition import TruncatedSVD
import numpy as np
from typing import Optional, List, Sequence, Tuple, Union

PCA_MAX_ENTRIES = 1000000

@dataclass
class GPParams:
    """
    Configures a Gaussian Process model for regression.

    Attributes:
        kernel: the kernel to use, either 'RBFKernel', 'MaternKernel', or
            'SpectralMixtureKernel'
        kernel_independent_scaling: whether to scale the kernel independently 
            for each input dimension (default True)
        kernel_num_mixtures: the number of mixtures to use in s
            SpectralMixtureKernel.
        kernel_smoothness (int): the smoothness of a MaternKernel (default 2)
        loss: 'marginal' or 'conditional'.  If 'marginal', the model is
            trained to maximize the marginal likelihood over the entire task.
            If 'conditional', the model is trained to minimize the conditional
            likelihood of the query set given the support set.
        PCA (bool): whether to use PCA (not supported yet)
        PCA_dim (int): the number of PCA dimensions
        multi_output (bool): whether to use a multi-output GP, which enables
            correlations between outputs. If false, each output is modeled
            as an independent GP.
        rank (int): the rank of the multitask kernel. If None, rank is the same as num outputs
        zero_mean (bool): whether to enforce a zero mean
        scale_to_bounds (bool): whether to scale the inputs to bounds [-1,1]
        noise_constraint (float or tuple, optional): If given, the noise is
            constrained to either a fixed value or an interval.  If a float, a
            FixedNoiseGaussianLikelihood is used.  If a tuple, this gives
            [lower,upper] bounds for a GaussianLikelihood.
    """
    kernel : str = 'RBFKernel'
    kernel_independent_scaling : bool = True
    kernel_num_mixtures : int = 4
    kernel_smoothness : int = 2
    loss : str = 'marginal'
    PCA : bool = False
    PCA_dim : int = 10
    multi_output : bool = False
    rank: Optional[int] = None 
    zero_mean: bool = False
    scale_to_bounds: bool = True
    noise_constraint : Optional[Union[float,Tuple[float,float]]] = None


def make_kernel(params : dict, num_input_dims : int = None):
    """Create a kernel from a dictionary of parameters"""
    independent_scaling = params.get('kernel_independent_scaling', True)
    ard_num_dims = num_input_dims if independent_scaling else None
    if params['kernel'] == 'RBFKernel':
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
    elif params['kernel'] =='MaternKernel':
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=params['kernel_smoothness']+0.5,ard_num_dims=ard_num_dims))
    elif params['kernel'] =='SpectralMixtureKernel':
        return gpytorch.kernels.SpectralMixtureKernel(num_mixtures=params['kernel_num_mixtures'], ard_num_dims=ard_num_dims)
    else:
        raise ValueError("[ERROR] the kernel '" + str(params['kernel']) + "' is not supported for regression, use 'RBKKernel', 'MaternKernel', or 'SpectralMixtureKernel'")
    
"""
#OLD CODE?
class _GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, params):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = make_kernel(params,inducing_points.shape[-1])
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        x = self.scale_to_bounds(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
"""
        
class _ExactGPModule(gpytorch.models.ExactGP):
    """1-D GP"""
    def __init__(self, train_x, train_y, likelihood, params):
        super().__init__(train_x, train_y, likelihood)
        if params.get('zero_mean',False):
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = make_kernel(params,train_x.shape[-1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def set_train_data(self,x,y,strict=False):
        super().set_train_data(x,y,strict=strict)


class _ExactIndependentGPModule(torch.nn.Module):
    """N-D GP assuming independence between observations"""
    def __init__(self, train_x, train_y, likelihood, params):
        super().__init__()
        num_outputs = train_y.shape[-1]
        self.num_outputs = num_outputs
        GP_likelihoods = [copy.deepcopy(likelihood) for i in range(num_outputs)]

        #if params['GP_model_type'] == 'exact':
        models = [_ExactGPModule(train_x,train_y[:,i],GP_likelihoods[i], params) for i in range(num_outputs)]
        # elif params['GP_model_type'] == 'approximate':
        #     self.GP_model = GPClassificationModel(train_x,params)
        self.channel_models = gpytorch.models.IndependentModelList(*models)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*[m.likelihood for m in models])

    def forward(self,x):
        return self.channel_models(*[x]*len(self.channel_models.models))

    def set_train_data(self,x,y):
        if x is None or len(x)==0:
            for i,GP in enumerate(self.channel_models.models):
                GP.train_data = None
                GP.train_targets = None
            return
        for i,GP in enumerate(self.channel_models.models):
            GP.set_train_data(x, y[:,i], strict=False)


class _ExactMOGPModule(gpytorch.models.ExactGP):
    """N-D GP assuming correlation between observations"""
    def __init__(self, train_x, train_y, likelihood, params):
        super().__init__(train_x, train_y, likelihood)
        num_outputs = train_y.shape[-1]
        self.num_outputs = num_outputs
        if params.get('zero_mean',False):
            self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ZeroMean(), num_tasks=num_outputs)
        else:
            self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_outputs)
        rank = min(num_outputs,params.get('rank',num_outputs))
        #should we use this instead?
        #self.kernel = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=num_outputs, rank=num_outputs)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_outputs, rank=rank)
        self.covar_module = make_kernel(params,train_x.shape[-1]).base_kernel  #.base_kernel: ignore scale kernel
        #self.covar_module = make_kernel(params,train_x.shape[-1])
        
        noise_constraint=None
        if hasattr(params,'noise_constraint'):
            if isinstance(params['noise_constraint'],(int,float)):
                noise_constraint = gpytorch.constraints.Interval(params['noise_constraint'], params['noise_constraint'])
            else:
                noise_constraint = gpytorch.constraints.Interval(params['noise_constraint'][0], params['noise_constraint'][1])
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs,noise_constraint=noise_constraint,rank=0,has_task_noise=True,has_global_noise=False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        mean_flat = mean_x.reshape(-1)  #flatten predictions
        #covar_x = self.covar_module(x)
        covar_x = to_linear_operator(self.covar_module.forward(x,x))
        #i = torch.arange(self.num_outputs)
        #covar_i = self.task_covar_module(i)
        covar_i = self.task_covar_module.covar_matrix
        if len(x.shape[:-2]):
            covar_i = covar_i.repeat(*x.shape[:-2], 1, 1)
        #covar_x = covar_x.evaluate()  #lazy tensor, need to convert it into matrix 
        covar = KroneckerProductLinearOperator(covar_x, covar_i)
        # if not self.training:
        #     print("x",x.shape)
        #     print("mean x",mean_x.shape)
        #     print("covar x",covar_x.shape)
        #     print("covar",covar.shape)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar, True, interleaved=True)
        return gpytorch.distributions.MultivariateNormal(mean_flat, covar)
    
    def forward_mean(self, x):
        """Similar to forward(), but only evaluates the mean.  Faster than forward()
        for prediction because the covariance matrix is not evaluated."""
        return self.mean_module(x) 
    
    def forward_variance(self, x):
        """Similar to forward(), but only evaluates the variance of each task.
        Faster than forward() because the covariance is not evaluated."""
        #tile covariance across all tasks
        covar_x = self.covar_module(x)
        var_x = torch.diag(covar_x)
        #repeat var_x for each task
        var_x_flat = var_x.repeat_interleave(self.num_outputs)
        #TODO: make this faster
        indices_all = torch.arange(self.num_outputs).repeat(x.shape[0])
        covar_i = self.task_covar_module(indices_all)
        var = var_x_flat.mul(torch.diag(covar_i))
        return var

    def set_train_data(self, x, y):
        if x is None:
            self.train_inputs = None
            self.train_targets = None
            return
        super().set_train_data(x, y, strict=False)


class GP(TorchModelModule):
    """
    GP model core implementation.

    Predicts backbone -> PCA -> GP_model -> reward prediction
    """
    def __init__(self, problem: SupervisedLearningProblem, params : dict):
        super(GP, self).__init__(problem)
        self.params = params
        x_dims = problem.core_input_shape()[0]
        y_dims = problem.core_output_shape()[0]
        self.PCA_matrix = None
        if params.get('PCA',False):
            train_x = torch.zeros((1,params['PCA_dim']))
        else:
            train_x = torch.zeros((1,x_dims))
        train_y = torch.zeros((1,y_dims))

        if params.get('scale_to_bounds',False):
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        else:
            self.scale_to_bounds = None

        #likelihood function
        if params.get('noise_constraint',None) is not None:
            if isinstance(params['noise_constraint'],tuple):
                noise_constraint=gpytorch.constraints.Interval(params['noise_constraint'][0], params['noise_constraint'][1])
                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
            else:
                assert isinstance(params['noise_constraint'],(int,float))
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise_constraint=params['noise_constraint'])
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if not params.get('multi_output',False):
            self.gmodel = _ExactIndependentGPModule(train_x,train_y,likelihood,params)
            self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.gmodel.likelihood,self.gmodel.channel_models)
            #self.mll = gpytorch.mlls.VariationalELBO(self.GP_likelihood, self.GP_model, num_data=len(train_loader.dataset))
        else:
            self.gmodel = _ExactMOGPModule(train_x,train_y,likelihood,params)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gmodel.likelihood, self.gmodel)
        self.conditioned_model = None

    def set_train_data(self, x, y):
        assert len(x) == len(y)
        if self.params.get('PCA',False):
            if self.PCA_matrix is None:
                raise RuntimeError("Need to set PCA matrix before calling set_train_data")
            x = torch.dot(x,self.PCA_matrix.T)
        if len(x)==0:
            self.gmodel.set_train_data(None,None)
            return
        if self.scale_to_bounds:
            x = self.scale_to_bounds(x)
        self.gmodel.set_train_data(x,y)
        #print("Set the GP model on the training data")

    @torch.no_grad()
    def condition_core(self, support_x, support_y):
        assert len(support_x) == len(support_y)
        self.eval()
        self.set_train_data(support_x, support_y)

    def forward_core(self, x):
        if self.params.get('PCA',False):
            if self.PCA_matrix is None:
                raise RuntimeError("Need to set PCA matrix before calling forward")
            x = torch.dot(x,self.PCA_matrix.T)
        if self.scale_to_bounds:
            x = self.scale_to_bounds(x)
        fpred = self.gmodel(x)  #returns a MultivariateNormal or list of MultivariateNormal objects
        if isinstance(fpred,list):
            Pypred = self.gmodel.likelihood(*fpred)
            ypred = torch.stack([p.mean for p in Pypred],dim=1)
            ystd = torch.stack([p.stddev for p in Pypred],dim=1)
        else:
            Pypred = self.gmodel.likelihood(fpred)
            ypred = Pypred.mean
            ystd = Pypred.stddev
        return ypred, ystd
        
    def forward_core_loss(self, x, y):
        if self.params.get('PCA',False):
            if self.PCA_matrix is None:
                raise RuntimeError("Need to set PCA matrix before calling forward")
            x = torch.dot(x,self.PCA_matrix.T)
        if self.scale_to_bounds:
            x = self.scale_to_bounds(x)
        fpred = self.gmodel(x)  #returns a MultivariateNormal or list of MultivariateNormal objects
        loss = -self.mll(fpred,y.T)   #expects the y to have shape (output_dim, batch_size)
        if isinstance(fpred,list):
            Pypred = self.gmodel.likelihood(*fpred)
            ypred = torch.stack([p.mean for p in Pypred],dim=1)
        else:
            Pypred = self.gmodel.likelihood(fpred)
            ypred = Pypred.mean
        return loss, ypred

    def forward_core_fewshot_loss(self, support_set, query_set):
        if self.params.get('PCA',False):
            if self.PCA_matrix is None:
                raise RuntimeError("Need to set PCA matrix before calling forward")
            raise NotImplementedError("Can't do PCA yet")
        support_x, support_y = support_set
        query_x, query_y = query_set
        if self.scale_to_bounds:
            support_x = self.scale_to_bounds(support_x)
            query_x = self.scale_to_bounds(query_x)
        if self.params.get('loss','marginal') == 'marginal':
            all_x = torch.concat((support_x,query_x),dim=-2)
            all_y = torch.concat((support_y,query_y),dim=-2)
            nsupp = support_x.shape[-2]
            self.set_train_data(all_x,all_y)
            l,p = self.forward_core_loss(all_x,all_y)
            return l,p[nsupp:]
        else:
            raise NotImplementedError("TODO: can't maximize conditional log likelihood yet")
            self.condition_core(support_x,support_y)      
            return self.forward_core_loss(query_x,query_y)

    def load_state_dict(self,states : dict):
        if 'PCA_matrix' in states:
            self.PCA_matrix = states['PCA_matrix']
            del states['PCA_matrix']
        super().load_state_dict(states)
    
    def state_dict(self) -> dict:
        res = super().state_dict()
        if self.PCA_matrix is not None:
            res['PCA_matrix'] = self.PCA_matrix
        return res


class GPModel(TorchTrainableModel,AdaptiveModel,ProbabilisticModel):
    """Gaussian Process model.

    Accepts multi-output prediction, and can be trained in few-shot or 
    standard supervised training manner.
    
    See :class:`GPParams` for a description of all configurable parameters.
    """
    def __init__(self, problem : SupervisedLearningProblem, params : Optional[Union[dict,GPParams]]=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
        if isinstance(params,GPParams):
            params = asdict(params)
        TorchTrainableModel.__init__(self,problem,params)
        self.model = GP(problem,params).to(self.device)   # type: GP
        self.reset()
                    
    def reset(self):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.Tensor([]).float().to(self.device)
        self.support_y = torch.Tensor([]).float().to(self.device)
        self.model.condition_core(self.support_x,self.support_y)
        
    def update(self,x,y):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.cat((self.support_x,self.problem.encode_input(totorch(x).unsqueeze(0).float().to(self.device))),0)
        self.support_y = torch.cat((self.support_y,self.problem.encode_output(totorch(y).unsqueeze(0).float().to(self.device))),0)
        assert len(self.support_x.shape)==2
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
            return super(GPModel,self).__call__(x)

    def train(self, train_data : MultiTaskDataset, val_data : Optional[MultiTaskDataset]=None, writer : Optional[SummaryWriter] = None):
        if self.model.params.get('PCA',False):
            #perform PCA on training data
            items = []
            for i in range(len(train_data)):
                for j in range(len(train_data[i])):
                    items.append(self.problem.encode_input(train_data[i][j]))
                    if len(items)*len(items[-1]) >= PCA_MAX_ENTRIES:
                        break
                if len(items)*len(items[-1]) >= PCA_MAX_ENTRIES:
                    break
            
            A = np.array(items)
            svd = TruncatedSVD(self.model.params['PCA_dim'])
            svd.fit(A)
            self.model.PCA_matrix = totorch(svd.components_)
        super(GPModel,self).train(train_data,val_data,writer)