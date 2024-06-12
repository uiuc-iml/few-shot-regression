import os, glob
from typing import Union,Optional,List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gpytorch

from ..data import MultiTaskDataset,FlatDataset,FewShotDataset
from ..utils import MLP,TorchTrainableModel, totorch, EarlyStopping
from ..model import SupervisedLearningProblem,AdaptiveModel,ProbabilisticModel,TorchModelModule
from ..GP.model import GP
from .. import utils

class DNNMeanResidualGPs(TorchModelModule):
    """
    Uses a DNN to model mean + a GP online to model residuals.

    Parameters:
        hidden_layers (list of int): # of hidden layers in fully-connected
            network
        hidden_batchnorm (bool): whether to batch-normalize in hidden layers.
        gp_kernel (str): RBFKernel or SpectralMixtureKernel
        gp_kernel_independent_scaling (bool): whether to independently scale
        
    """
    def __init__(self, problem : SupervisedLearningProblem, params : dict):
        super(DNNMeanResidualGPs, self).__init__(problem)
        self.params = params
        n_inputs = problem.core_input_shape()[0]
        n_outputs = problem.core_output_shape()[0]

        # Mean is used for offline training
        hidden_layers = params.get('hidden_layers',[4])
        hidden_batchnorm = params.get('hidden_batchnorm',False)
        self.mean_model = MLP([n_inputs]+hidden_layers+[n_outputs],batchnorm=hidden_batchnorm)
        self.mean_loss_fn = nn.MSELoss()

        #move all gp_ parameters to gp_params
        gp_params = {}
        for (k,v) in params.items():
            if k.startswith('gp_'):
                gp_params[k[3:]] = v
        gp_params['zero_mean'] = True  #force the MLP to absorb the constant offset?
        self.residual_model = GP(problem, gp_params)
        self.gp_disabled = False

    def disable_gp(self):
        self.gp_disabled = True
    
    def enable_gp(self):
        self.gp_disabled = False

    def forward_core(self, x):
        mean = self.mean_model(x)
        std = torch.zeros_like(mean)
        if not self.gp_disabled:
            res_mean, res_std = self.residual_model(x)
            return res_mean + mean, res_std
        else:
            return mean, std

    def forward_core_loss(self, x, y):
        y_pred, std = self.forward_core(x)
        return self.mean_loss_fn(y, y_pred), y_pred

    @torch.no_grad()    
    def condition_core(self, support_x, support_y):
        if len(support_x) == 0:
            self.residual_model.condition(support_x, support_y)
            return
        reward_mean_supp = self.mean_model(support_x)
        self.residual_model.condition(support_x, support_y - reward_mean_supp)
    
    def state_dict(self):
        checkpoint = dict()
        checkpoint['mean_model'] = self.mean_model.state_dict()
        checkpoint['residual_model'] = self.residual_model.state_dict()
        return checkpoint

    def load_state_dict(self, state):
        self.mean_model.load_state_dict(state['mean_model'])
        self.residual_model.load_state_dict(state['residual_model'])
    
class DNNResidualGPModel(TorchTrainableModel,AdaptiveModel,ProbabilisticModel):
    """
    Uses a DNN to model mean + a GP online to model residuals.

    GP kernel can be trained in standard or few-shot fashion.

    Configurable parameters:
    
        - hidden_layers (list of int): # of hidden layers in fully-connected
            network
        - hidden_batchnorm (bool): whether to batch-normalize in hidden layers.
        - optimizer (str): only supports Adam for now.
        - lr (float): learning rate
        - gp_k_shot (int or list, optional): if given, uses few-shot training
            with this # of draws to tune the kernel.  Otherwise, uses standard
            training.
        - gp_loss (str): 'marginal' or 'conditional'. Whether to use MLL over
            the entire task or just the query set conditioned on the support
            set. See GPModel for more information.
        - gp_kernel (str): rbf or spectral
        - gp_kernel_independent_scaling (bool): whether to independently scale
            each dimension of the GP input during learning (default true)
        - gp_lr (float): kernel learning rate
        - gp_training_iter (int): # of iters to train GP kernel
        - gp_PCA (bool): whether to use PCA (not supported yet)
        - gp_PCA_dim (int): # of PCA dimensions
        
        - checkpoint_dir (str): directory where training checkpoints are stored
        - num_workers (int): # of workers for dataloader
        - val_freq (int): # of steps between validation runs
        - batch_size (int): training batch size
        - epochs (int): # of epochs for offline training
    
    """
    def __init__(self, problem : SupervisedLearningProblem, params : Union[str,dict]=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
        TorchTrainableModel.__init__(self,problem,params)
        self.model = DNNMeanResidualGPs(problem,self.params).to(self.device)

    def train_step(self,x,y):
        self.model.disable_gp()
        loss, p = self.model.forward_loss(x,y)
        self.model.enable_gp() 
        return loss,p

    def test_step(self,x,y):
        return self.model.forward_loss(x,y)
        
    def gp_train_loop(self, gp_dataset):
        params = self.params
        print("GPs' training starts")
        self.model.mean_model.eval()
        self.model.residual_model.train()

        if params.get('gp_k_shot',None):
            gp_dataset = FewShotDataset(gp_dataset,params['gp_k_shot'],num_draws_per_task=1)
            gp_train_loader = DataLoader(gp_dataset,batch_size=1)
        else:
            gp_dataset = FlatDataset(gp_dataset)
            gp_train_loader = DataLoader(gp_dataset, batch_size=len(gp_dataset))

        optimizer = torch.optim.Adam(self.model.residual_model.parameters(), lr=params.get('gp_lr',0.1) )
        training_iter = params.get('gp_training_iter',500)

        if isinstance(gp_dataset,FewShotDataset):
            #do several iterations of few-shot training
            nsteps_task = int(np.ceil(training_iter / len(gp_train_loader)))
            def optimize_step(support_data, query_data):
                support_data = tuple([data.float().to(self.device) for data in support_data])
                query_data = tuple([data.float().to(self.device) for data in query_data])
                supp_x, supp_y = support_data
                query_x, query_y = query_data
               

                #for j in range(nsteps_task):
                optimizer.zero_grad()
                preds = []
                sumloss = None
                for b in range(len(supp_x)):
                    self.model.condition(supp_x[b],supp_y[b])
                    with torch.no_grad():
                        query_x_core = self.problem.encode_input(query_x)
                        query_y_core = self.problem.encode_output(query_y) - self.model.mean_model(query_x_core)
                    loss, p = self.model.residual_model.forward_core_loss(query_x_core[b],query_y_core[b])
                    if sumloss is None:
                        sumloss = loss
                    else:
                        sumloss += loss
                    preds.append(p)
                sumloss.backward(retain_graph=True)
                self.optimizer.step()
                return sumloss/b, torch.stack(preds,dim=0)  #aggregate predictions

            for epoch in range(training_iter):
                utils.fewshot_training_epoch(gp_train_loader, optimize_step)

        else:
            if len(gp_train_loader.dataset) > 2000:
                print("WARNING: lots of data in GP training... will take forever?")
                #input()
            self.model.condition(torch.Tensor([]).float().to(self.device),torch.Tensor([]).float().to(self.device))

            def optimize_step(x,y):
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                x_core = self.problem.encode_input(x)
                
                #labels = torch.from_numpy(labels).to(self.device)
                with torch.no_grad():
                    y_core_preds = self.model.mean_model(x_core)

                #for i in range(training_iter):
                optimizer.zero_grad()
                GP_loss, output = self.model.residual_model.forward_core_loss(x_core,y-y_core_preds)
                GP_loss.backward(retain_graph=True)
                optimizer.step()
                #    print('Iters %d/%d - Loss: %.3f' % (i + 1, training_iter, GP_loss.item()))
                return GP_loss, output
            
            for epoch in range(training_iter):
                utils.standard_training_epoch(gp_train_loader, optimize_step) # We want optimize_step function to receive the whole out
        self.model.residual_model.eval()
        print("GPs' training is finished")

    def train(self, train_data : MultiTaskDataset, val_data : Optional[MultiTaskDataset]=None):
        """Override TrainableModel"""
        writer = SummaryWriter(log_dir=self.params.get('log_dir','training_logs'))
        
        #train the mean using the MSE loss
        TorchTrainableModel.train(self, train_data, val_data, writer)

        fn = os.path.join(self.params['checkpoint_dir'], 'model.tar')      
        self.save(fn)
        #Now train the GP
         
        self.load(fn)  # The loaded model only has trained mean
        self.gp_train_loop(train_data)

        self.model.eval()
        self.save(fn)
        
        writer.flush()
        writer.close()

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
        """Override ProbabilisticScoringModel"""
        with gpytorch.settings.fast_pred_var():
            return super(DNNResidualGPModel, self).__call__(x)
            
            
