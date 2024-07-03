import os, glob
import random
import yaml
from typing import Union,Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gpytorch
import copy
from ..data import StandardMultiTaskDataset,MultiTaskDataset,FlatDataset,FewShotDataset,MergedMultiTaskDataset,SubsetMultiTaskDataset
from ..utils import MLP,TorchTrainableModel, TwoStageTorchTrainableModel,totorch, EarlyStopping, set_seed, to_device
from ..model import AdaptiveModel,ProbabilisticModel, SupervisedLearningProblem, TorchModelModule
from .. import utils
from dataclasses import dataclass,asdict,replace

class InTaskSplittingFlatDataset(FlatDataset):
    """Train-val split for a flatdataset"""
    def __init__(self,data : MultiTaskDataset, train_split = False, val_split = False, train_fraction = 0.9):
        self.all_data = []
        if train_split:
            for task in data:
                N = len(task)
                for i in range(0,int(train_fraction*N)):
                    self.all_data.append(task[i])
            return
        if val_split:
            for task in data:
                N = len(task)
                for i in range(int(train_fraction*N),N):
                    self.all_data.append(task[i])
            return

        for task in data:
            for i in range(len(task)):
                self.all_data.append(task[i])

    def __len__(self):
        return len(self.all_data)#sum(len(task) for task in self.data)
    
    def __getitem__(self,i:int):
        return self.all_data[i]

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='RBFKernel', params = None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        #self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        ard_num_dims = params.get('kernel_ard_num_dims', params['gp_input_dim'])
        if(kernel=='rbgKernel' or kernel=='RBFKernel'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        elif(kernel=='spectralKernel'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, \
                ard_num_dims = ard_num_dims)
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        x = self.scale_to_bounds(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepKernel(TorchModelModule):
    def __init__(self, problem : SupervisedLearningProblem, params : Union[str,dict]=None, device = 'cuda'):
        super(DeepKernel, self).__init__(problem)
        self.problem = problem
        self.params = params
        self.device = device

        # Mean is used for offline training
        hidden_layers = params.get('hidden_layers',[4])
        hidden_batchnorm = params.get('hidden_batchnorm',False)
        nonlinear = self.params.get('nonlinear', 'ReLU')

        self.mean_model = MLP([problem.core_input_shape()[0]]+hidden_layers+[self.problem.core_output_shape()[0]],batchnorm=hidden_batchnorm,\
            dropout=params.get('dropout', 0), nonlinear=nonlinear)
        self.lr = params['lr']
        self.mean_loss_fn = nn.MSELoss()

        if not params.get('use_noise_constraint', False):
            self.GP_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        else:
            self.GP_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.Interval(1., 25.))

        latent_dim = params['gp_input_dim']
        train_x=torch.ones(1, latent_dim)
        train_y=torch.ones(1)
        self.gp_model = ExactGPModel(train_x=train_x, train_y=train_y, likelihood=self.GP_likelihood, kernel=params['gp_kernel'], params=params)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.GP_likelihood, self.gp_model)

        self.batch_idx = 0

        GP_layers = params.get('gp_layers', [64,32,16])
        GP_use_batchnorm = params.get('gp_batchnorm', False)
        GP_nonlinear= self.params.get('gp_nonlinear', 'ReLU')
        self.deep_kernel = MLP([problem.core_input_shape()[0]] + GP_layers +\
            [params['gp_input_dim']], batchnorm=GP_use_batchnorm, nonlinear=GP_nonlinear)

        ## Make a copy of the random intial mean module weights for consistency of reinitialization
        self.init_mean_model_state_dict = copy.deepcopy(self.mean_model.state_dict())
        self.init_problem_x_encoder = None
        self.init_problem_y_encoder = None
        if self.problem.x_encoder is not None:
            self.init_problem_x_encoder = copy.deepcopy(self.problem.x_encoder.state_dict())
        if self.problem.y_encoder is not None:
            self.init_problem_y_encoder = copy.deepcopy(self.problem.y_encoder.state_dict())

        self.gp_disabled = False

    def disable_gp(self):
        self.gp_disabled = True
    
    def enable_gp(self):
        self.gp_disabled = False

    def forward_core(self, x):
        mean = self.mean_model(x)
        std = torch.zeros_like(mean)
        if not self.gp_disabled:
            x = self.deep_kernel(x)
            if len(self.support_x) == 0:
                res_mean= torch.zeros(mean.shape).to(self.device)
                res_std = torch.sqrt(self.gp_model.likelihood.noise_covar.noise)*torch.ones(mean.shape).to(self.device)
            else:    
                output_gp = self.gp_model(x)
                dist = self.GP_likelihood(output_gp)
                res_mean, res_std = dist.mean, dist.stddev
                res_std = res_std.unsqueeze(-1)
            return res_mean + mean, res_std
        else:
            return mean, std

    def forward_core_loss(self, x, y):
        y_pred, std = self.forward_core(x)
        return self.mean_loss_fn(y, y_pred), y_pred

    def non_GP_state_dict(self):
        ''' Return all state dict but GP'''
        checkpoint = dict()
        checkpoint['mean'] = self.mean_model.state_dict()
        if self.init_problem_x_encoder is not None:
            checkpoint['x_encoder'] = self.problem.x_encoder.state_dict()
        else:
            checkpoint['x_encoder'] = None
        if self.init_problem_y_encoder is not None:
            checkpoint['y_encoder'] = self.problem.y_encoder.state_dict()
        else:
            checkpoint['y_encoder'] = None
        return checkpoint

    def load_non_GP_state_dict(self, state):
        self.mean_model.load_state_dict(state['mean'])
        if self.init_problem_x_encoder is not None:
            self.problem.x_encoder.load_state_dict(state['x_encoder'])
        if self.init_problem_y_encoder is not None:
            self.problem.y_encoder.load_state_dict(state['y_encoder'])

    @torch.no_grad()    
    def condition_core(self, support_x, support_y):
        self.support_x = support_x
        if len(support_x) == 0:
            self.gp_model.set_train_data(None, None, strict = False)
            return
        reward_mean_supp = self.mean_model(support_x)
        support_x = self.deep_kernel(support_x)
        self.gp_model.set_train_data(support_x, (support_y - reward_mean_supp).squeeze(-1), strict = False)

    def reinitialize_mean_model(self):
        self.mean_model.load_state_dict(self.init_mean_model_state_dict)
   
    def reintialize_problem_encoders(self):
        if self.init_problem_x_encoder is not None:
            self.problem.x_encoder(self.init_problem_x_encoder)
        if self.init_problem_y_encoder is not None:  
            self.proboem.y_encoder(self.init_problem_y_encoder)


class CoDeGaModel(TwoStageTorchTrainableModel,AdaptiveModel,ProbabilisticModel):
    """
    Configurable parameters:
    
        hidden_layers (list of int): # of hidden layers in fully-connected
            network
        hidden_batchnorm (bool): whether to batch-normalize in hidden layers.
        optimizer (str): only supports Adam for now.
        lr (float): learning rate
        gp_k_shot (int or list, optional): if given, uses few-shot training with this
            # of draws to tune the kernel.  Otherwise, uses standard training.
        gp_kernel (str): rbf or spectral
        gp_lr (float): kernel learning rate
        gp_training_iter (int): # of iters to train GP kernel
        
        checkpoint_dir (str): directory where training checkpoints are stored
        num_workers (int): # of workers for dataloader
        val_freq (int): # of steps between validation runs
        batch_size (int): training batch size
        epochs (int): # of epochs for offline training
    
    """
    def __init__(self, problem : SupervisedLearningProblem, params : Union[str,dict]=None, benchmark=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
        TorchTrainableModel.__init__(self,problem,params)
        self.model = DeepKernel(problem,self.params, self.device).to(self.device)
        self.batch_idx = 0
        # Note GP_model has a separate optimizer in gp_train_loop

    def train_epoch(self, dataloader, epoch, writer):
        return self.mean_train_epoch(dataloader, epoch, writer)

    def mean_train_epoch(self, dataloader, epoch, writer = None, penalize_encoder = False):
        self.model.train()
        LEARNING_RATE_CLIP = 1e-5 
        for g in self.optimizer.param_groups:
            g['lr'] = max(self.params['lr'] * (self.params['lr_decay'] ** (epoch // self.params['lr_step'])), LEARNING_RATE_CLIP)

        def train_step(x,y):
            self.optimizer.zero_grad()
            x = to_device(x,self.device,float)
            y = to_device(y,self.device,float)
            loss, p = self.model.forward_loss(x,y)         
            if penalize_encoder:
                penalty = self.params['split_encoder_penalty']*sum([((p1-p2)**2).sum() for (p1,p2) in zip(self.task.encoder_x.parameters(), \
                    self.encoder_x_weights)]) 
                loss += penalty     
            loss.backward(retain_graph=True)
            self.optimizer.step()
            return loss,p
        utils.standard_training_epoch(dataloader, train_step,
            writer, epoch)

    def mean_val_epoch(self, dataloader, epoch, writer):
        self.model.eval()
        def test_step(x,y):
            x = to_device(x,self.device,float)
            y = to_device(y,self.device,float)
            return self.model.forward_loss(x,y)
        return utils.standard_testing_epoch(dataloader, test_step,
                            writer=writer,epoch=epoch)

    def zero_shot_train(self, train_loader, val_loader, writer = None):
        ## first train DNN with all the data. Then the backbone will be frozen.
        patience = self.params.get('patience',10)
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)     
        self.model.train()
        self.model.disable_gp() 
        for epoch in range(self.params['epochs']):
            self.mean_train_epoch(train_loader, epoch, writer) 
            res = self.mean_val_epoch(val_loader,epoch,writer)
            valid_loss = res[0]
            self.early_stopping(valid_loss, self)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        self.model.enable_gp()

    def adaptive_train(self, train_data, val_data):
        # controlled splits training
        all_data = MergedMultiTaskDataset(train_data, val_data)
        all_task_indices = [i for i in range(len(all_data))]

        ## load terrain ID splits and create all data indices k-fold splits
        guided_splits = self.params.get('guided_splits', None)
        if guided_splits is None:
            #5 random splits by default
            for _ in range(5):
                task_indices_copy = copy.copy(all_task_indices)
                indices_list.append(random.shuffle(task_indices_copy)[0:int(len(task_indices_copy)/2)])
        else:   
            if isinstance(guided_splits, str):
                guided_splits = list(np.load(guided_splits, allow_pickle=True))
        indices_list = guided_splits

        # mean split training
        GP_train_data_lists = []
        for idx, indices in enumerate(indices_list):  
            print(f'----------------- Fold {idx} Training ---------------')
            mean_indices = list(set(all_task_indices) - set(indices))
            GP_train_data = SubsetMultiTaskDataset(all_data, indices)
            mean_data = SubsetMultiTaskDataset(all_data, mean_indices)
            mean_train_dataset = InTaskSplittingFlatDataset(mean_data, train_split= True)
            mean_val_dataset = InTaskSplittingFlatDataset(mean_data, val_split=True)
            mean_train_loader = DataLoader(dataset=mean_train_dataset, batch_size=self.params.get('adaptive_batch_size', 64),\
                shuffle=True, num_workers=self.params.get('num_workers',0))
            DNN_val_loader =  DataLoader(dataset=mean_val_dataset, batch_size=self.params.get('adaptive_batch_size', 64),\
                shuffle=False, num_workers=self.params.get('num_workers',0))
            
            self.model.reinitialize_mean_model()
            self.model.reintialize_problem_encoders()
            self.zero_shot_train(mean_train_loader, DNN_val_loader, writer = None)
            GP_train_data_lists.append(GP_train_data)
            # save weights for batch training later 
            fn = os.path.join(self.params['checkpoint_dir'], f'fold_{idx}_DNN_and_backbone_model.tar')     
            torch.save(self.model.non_GP_state_dict(),fn)

        print('-------- Train Deep Kernels ------------')
        epochs = self.params.get('CoDeGa_gp_training_epoch', 200)
        min_total_loss = 1e6
        min_total_loss_epoch = 0
        patience = self.params.get('CoDeGa_patience', 5)
        
        for epoch in range(epochs):
            print(f'cross val batch training, epoch {epoch}')
            with open(os.path.join(self.params['checkpoint_dir'],'GP_epoch.txt'), 'w') as f:
                f.write('%d' % epoch)
            lr = max(self.params['gp_lr'] * (self.params['gp_lr_decay'] ** (epoch // self.params['gp_lr_step'])), 1e-5)
            total_loss = 0.
            total_data = 0

            inds = np.arange(len(GP_train_data_lists))
            np.random.shuffle(inds)
            for idx in inds:
                GP_train_data = GP_train_data_lists[idx]
                fn = os.path.join(self.params['checkpoint_dir'], f'fold_{idx}_DNN_and_backbone_model.tar') 
                checkpoint = torch.load(fn)
                self.model.load_non_GP_state_dict(checkpoint)
                loss = self.gp_train_loop(GP_train_data, epoch, lr, True)
                total_loss += loss*len(GP_train_data)
                total_data += len(GP_train_data)

            total_loss = total_loss/total_data
            if total_loss < min_total_loss:
                min_total_loss = total_loss
                min_total_loss_epoch = epoch
            else:
                if epoch - min_total_loss_epoch >= patience:
                    print('Deep kernel training early stopping')
                    break

    def gp_train_loop(self, gp_dataset, epoch = 0, lr = None, return_loss = False):
        params = self.params
        self.model.train()
        self.model.mean_model.eval()
        if self.problem.x_encoder is not None:
            self.problem.x_encoder.eval()
        if self.problem.y_encoder is not None:
            self.problem.y_encoder.eval()

        gp_dataset = FewShotDataset(gp_dataset, params['gp_k_shot'], num_draws_per_task=1)#num_draws need to be 1 for splitting
        gp_train_loader = DataLoader(gp_dataset,batch_size=1)

        if lr is None:
            lr = params.get('gp_lr',0.1)

        optimizer = torch.optim.Adam([{'params': self.model.gp_model.parameters()},
                                    {'params': self.model.deep_kernel.parameters()},],lr = lr)
        
        #do several iterations of few-shot training
        #nsteps_task = int(np.ceil(training_iter / len(gp_train_loader)))
        def optimize_step(support_data, query_data):
            LEARNING_RATE_CLIP = 1e-5
            for g in optimizer.param_groups:
                g['lr'] = max(self.params['gp_lr'] * (self.params['gp_lr_decay'] ** (epoch // self.params['gp_lr_step'])), LEARNING_RATE_CLIP)
            optimizer.zero_grad()

            with torch.no_grad():
                if len(support_data) > 2: # not simply [x,y]
                    x_core_supp = self.problem.encode_input(tuple([data.float().to(self.device) for data in support_data[:-1]]))
                    x_core_query = self.problem.encode_input(tuple([data.float().to(self.device) for data in query_data[:-1]]))
                    y_core_supp = self.problem.encode_output(tuple([data.float().to(self.device) for data in support_data[-1]]))
                    y_core_query = self.problem.encode_output(tuple([data.float().to(self.device) for data in query_data[-1]]))
                else:
                    x_core_supp = self.problem.encode_input(support_data[0].float().to(self.device))
                    x_core_query = self.problem.encode_input(query_data[0].float().to(self.device))
                    y_core_supp = self.problem.encode_output(support_data[1].float().to(self.device))
                    y_core_query = self.problem.encode_output(query_data[1].float().to(self.device))
                support_N = x_core_supp.size(-2)
                x_core = torch.cat([x_core_supp, x_core_query], dim=-2)
                y_core = torch.cat([y_core_supp, y_core_query], dim=-2)
            
            preds = []
            sumloss = None
            for b in range(len(x_core)):
                with torch.no_grad():
                    mean = self.model.mean_model(x_core[b])
                    self.model.condition_core(x_core[b],y_core[b])
                    residuals = y_core - mean
                x_gp = self.model.deep_kernel(x_core[b])
                output_gp = self.model.gp_model(x_gp)
                dist = self.model.GP_likelihood(output_gp)
                gp_loss = -self.model.mll(output_gp,residuals.squeeze(-1)) 
                p = dist.mean.unsqueeze(-1) + mean
                if sumloss is None:
                    sumloss = gp_loss
                else:
                    sumloss += gp_loss

                preds.append(p[support_N:,:])
            sumloss.backward(retain_graph = True)
            optimizer.step()
            return sumloss/(b+1), torch.stack(preds,dim=0)

        GP_loss = utils.fewshot_training_epoch(gp_train_loader, optimize_step, return_loss = return_loss)
        if return_loss:
            return GP_loss

    def train(self, train_data : MultiTaskDataset, val_data : Optional[MultiTaskDataset]=None, writer : Optional[SummaryWriter] = None):
        """overwriting because of the controlled splits training"""
        import time
        from shutil import copyfile

        torch.autograd.set_detect_anomaly(True) # detects nans and terminates training
        if self.problem.train_on_encoded_x or self.problem.train_on_encoded_y:
            print("Training on encoded input and/or output")
            #create a new dataset of encoded input/output pairs and train on that
            x_encoder = self.problem.x_encoder if not self.problem.train_on_encoded_x else None
            y_encoder = self.problem.y_encoder if not self.problem.train_on_encoded_y else None
            mod_problem = replace(self.problem,x_encoder=x_encoder,y_encoder=y_encoder,train_on_encoded_x=False,train_on_encoded_y=False)
            orig_problem = mod_problem
            self.problem = mod_problem
            train_tasks = [[(self.problem.encode_input(x),self.problem.encode_output(y)) for x,y in task] for task in train_data]
            train_data = StandardMultiTaskDataset(train_tasks)
            if val_data is not None:
                val_tasks = [[(self.problem.encode_input(x),self.problem.encode_output(y)) for x,y in task] for task in val_data]
                val_data = StandardMultiTaskDataset(val_tasks)
            self.train(train_data,val_data)           
            self.problem = orig_problem
            return

        # optimizer for mean function
        if self.optimizer is None:
            if self.params['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}],lr = self.params['lr'])
            else:
                raise ValueError('Unknown optimization, please define by yourself')

        # copy config file, helps in reproducibility
        checkpoint_dir = self.params.get('checkpoint_dir',None)
        if checkpoint_dir is not None:
            os.makedirs(self.params['checkpoint_dir'], exist_ok=True)
            if self.params_file is not None:
                copyfile(self.params_file, os.path.join(self.params['checkpoint_dir'], 'config.yaml'))
            else:
                with open(os.path.join(self.params['checkpoint_dir'],'config.yaml'),'w') as f:
                    yaml.dump(self.params,f)

        if self.params.get('seed',None) is not None:
            set_seed(self.params['seed'])
        created_writer = False
        if writer is None:
            writer = SummaryWriter(log_dir=self.params.get('log_dir','training_logs'))
            created_writer = True

        zero_shot_batch_size = self.params.get('zero_shot_batch_size',None)
        zero_shot_num_workers = self.params.get('zero_shot_num_workers',0)
        zero_shot_train_dataset = FlatDataset(train_data)
        print("Using a zero-shot train dataset of length",len( zero_shot_train_dataset))
        if zero_shot_batch_size is not None:
            zero_shot_train_loader = DataLoader(dataset= zero_shot_train_dataset, batch_size= zero_shot_batch_size, shuffle=True, num_workers= zero_shot_num_workers)#, pin_memory=True)
        else:
            zero_shot_train_loader = DataLoader(dataset= zero_shot_train_dataset, batch_size=len( zero_shot_train_dataset), shuffle=True, num_workers= zero_shot_num_workers)#, pin_memory=True)
            
        # set batch_size  and num_workers to 1 and keep them fixed for consistency across experiments
        zero_shot_val_loader = None
        if val_data is not None:
            st = time.time()
            zero_shot_val_dataset = FlatDataset(val_data)
            if  zero_shot_batch_size is not None:
                zero_shot_val_loader = DataLoader(dataset= zero_shot_val_dataset, batch_size= zero_shot_batch_size, shuffle=False, num_workers= zero_shot_num_workers)#, pin_memory=True)
            else:
                zero_shot_val_loader = DataLoader(dataset= zero_shot_val_dataset, batch_size=len( zero_shot_val_dataset), shuffle=False, num_workers= zero_shot_num_workers)#, pin_memory=True)

        # mean training and saving
        print('-------- Train DNN on all data first ------------')
        self.zero_shot_train(zero_shot_train_loader, zero_shot_val_loader, writer)
        mean_fn = os.path.join(self.params['checkpoint_dir'], 'DNN_and_backbone_model.tar')   
        torch.save(self.model.non_GP_state_dict(),mean_fn) #Save a copy of trained parameters for loading later
        if self.params.get('penalize_split_encoder', False):
            self.x_encoder_weights = copy.deepcopy(self.problem.x_encoder.parameters())
        
        # kernel training 
        self.adaptive_train(train_data, val_data)
        self.model.eval()

        # Finally reload the trained mean and save it
        checkpoint = torch.load(mean_fn)
        self.model.load_non_GP_state_dict(checkpoint)

        if hasattr(self,'reset'):
            #during training, some support sets may have been set. These need to be cleared
            self.reset()

        if created_writer:
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
            return super(CoDeGaModel, self).__call__(x)
            