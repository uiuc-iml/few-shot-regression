#modified based on https://github.com/aravindr93/imaml_dev
import os, glob
from typing import Union,Optional,List

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('..')
from torch.nn import Linear, Conv2d, Sequential, LeakyReLU, MaxPool2d, LocalResponseNorm, BatchNorm1d
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import gpytorch
import os, copy

import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ..model import SupervisedLearningProblem,AdaptiveModel,ProbabilisticModel,TorchModelModule
from ..data import MultiTaskDataset,FewShotDataset, MergedMultiTaskDataset
from ..utils import MLP, TorchTrainableModel,weights_init, totorch, to_device, fromtorch,tobatch,frombatch,get_torchmodel_params,set_torchmodel_params
from .. import utils

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
        
    def forward(self, x):
        return x


class iMAML(TorchModelModule,nn.Module):
    def __init__(self, problem : SupervisedLearningProblem, params : dict, is_fast_model = False):
        nn.Module.__init__(self)
        TorchModelModule.__init__(self, problem)
        self.problem = problem
        self.params = params
        self.is_fast_model = is_fast_model
        if self.is_fast_model:
            self.x_encoder = copy.deepcopy(self.problem.x_encoder)
            self.y_encoder = copy.deepcopy(self.problem.y_encoder)
        hidden_layers = params.get('hidden_layers',[4])
        hidden_batchnorm = params.get('hidden_batchnorm',False)
        nonlinear = self.params.get('nonlinear', 'ReLU')
        self.MLP = MLP([problem.core_input_shape()[0]]+hidden_layers+[self.problem.core_output_shape()[0]],\
            batchnorm=hidden_batchnorm,dropout=params.get('dropout', 0), nonlinear=nonlinear)
        self.loss_fn = nn.MSELoss()


    def fast_model_forward_loss(self, x, y):
        return self.forward_core_loss(self.x_encoder(x), self.y_encoder(y))

    def forward_core(self, x):
        mean = self.MLP(x)
        std = torch.zeros_like(mean)
        return mean, std

    def condition_core(self, support_x, support_y):
        if len(support_x) == 0:
            return

        for g in self.optimizer.param_groups:
            g['lr'] = self.params['online_lr']

        self.train()
        num_epochs = self.params.get('online_train_epochs',1)
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            loss, p = self.forward_core_loss(support_x,support_y)
            loss.backward()
            self.optimizer.step()
        self.finetuned = True
        self.eval()
        
    def forward_core_loss(self, x, y):
        y_pred, std = self.forward_core(x)
        return self.loss_fn(y, y_pred), y_pred

    def forward_fewshot_loss(self,support_set,query_set):
        support_x, support_y = support_set
        query_x, query_y = query_set
        support_x, support_y  = (self.problem.encode_input(support_x), self.problem.encode_output(support_y))
        query_x, query_y = (self.problem.encode_input(query_x), self.problem.encode_output(query_y))
        self.condition_core(support_x,support_y)
        return self.forward_core_loss(query_x,query_y)

    def regularization_loss(self, initial_x_encoder_weights, \
            initial_y_encoder_weights, initial_weights, lam = 1.0):
        offset = 0
        regu_loss = 0.5 * lam * sum([((p1-p2)**2).sum() for (p1,p2) in zip(self.MLP.parameters(), \
            initial_weights)]) 
        if self.is_fast_model:
            regu_loss += 0.5 * lam * sum([((p1-p2)**2).sum() for (p1,p2) in \
                zip(self.x_encoder.parameters(), initial_x_encoder_weights)]) 
            regu_loss += 0.5 * lam * sum([((p1-p2)**2).sum() for (p1,p2) in \
                zip(self.y_encoder.parameters(), initial_y_encoder_weights)]) 
        else:
            regu_loss += 0.5 * lam * sum([((p1-p2)**2).sum() for (p1,p2) in \
                zip(self.problem.x_encoder.parameters(), initial_x_encoder_weights)]) 
            regu_loss += 0.5 * lam * sum([((p1-p2)**2).sum() for (p1,p2) in \
                zip(self.problem.y_encoder.parameters(), initial_y_encoder_weights)]) 
        return regu_loss
        
    def matrix_evaluator(self, data, lam, regu_coef=1.0, lam_damping=100.0):
        """
        Constructor function that can be given to CG optimizer
        
        """
        # if type(lam) == np.ndarray:
        #     lam = utils.to_device(lam, self.use_gpu)
        def evaluator(v):
            hvp = self.hessian_vector_product(data, v) #Returns \nabla^2_\phi \hat(L)(\phi) * v 
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping) # performs (I + \nabla^2_\phi \hat(L)(\phi)) * v  with some damping stuff
            return Av
        return evaluator
    
    def hessian_vector_product(self, data, vector):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        tloss, p = self.fast_model_forward_loss(data[0], data[1])
        if self.is_fast_model:
            grad_ft = torch.autograd.grad(tloss, list(self.x_encoder.parameters()) + \
                list(self.y_encoder.parameters()) + \
                list(self.MLP.parameters()), create_graph=True)
        else:
            grad_ft = torch.autograd.grad(tloss, list(self.problem.x_encoder.parameters()) + \
                list(self.problem.y_encoder.parameters()) + \
                list(self.MLP.parameters()), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        # vec = utils.to_device(vector, self.use_gpu)
        h = torch.sum(flat_grad * vector)
        if self.is_fast_model:
            hvp = torch.autograd.grad(h, list(self.x_encoder.parameters()) + \
                list(self.y_encoder.parameters()) + \
                list(self.MLP.parameters()))
        else:
            hvp = torch.autograd.grad(h,list(self.problem.x_encoder.parameters()) + \
                list(self.problem.y_encoder.parameters()) + \
                list(self.MLP.parameters()))
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat
    
    def cg_solve(self, f_Ax, b, cg_iters=10, verbose=False, residual_tol=1e-10, x_init=None):
        """
        Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
        Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
        Algorithm template from wikipedia
        Verbose mode works only with numpy
        """
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()

        fmtstr = "%10i %10.3g %10.3g %10.3g"
        titlestr = "%10s %10s %10s %10s"
        if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

        for i in range(cg_iters):
            rdotr = r.dot(r)
            Ap = f_Ax(p)
            alpha = rdotr/(p.dot(Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            newrdotr = r.dot(r)
            beta = newrdotr/rdotr
            p = r + beta * p
            if newrdotr < residual_tol:
                break

        return x

    def outer_step_with_grad(self, grad, optimizer, weights):
        """
        Given the gradient, step with the outer optimizer using the gradient.
        Assumed that the gradient is a tuple/list of size compatible with model.parameters()
        If flat_grad, then the gradient is a flattened vector
        """
        check = 0
        for p in self.MLP.parameters():
            check = check + 1 if type(p.grad) == type(None) else check
        if check > 0:
            # initialize the grad fields properly
            dummy_loss = self.regularization_loss(weights[0], weights[1], weights[2])
            dummy_loss.backward()  # this would initialize required variables

        offset = 0
        if self.is_fast_model:
            for p in self.x_encoder.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad.copy_(this_grad)
                offset += p.nelement()
            for p in self.y_encoder.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad.copy_(this_grad)
                offset += p.nelement()
        else:
            for p in self.problem.x_encoder.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                # print(this_grad)
                p.grad.copy_(this_grad)
                offset += p.nelement()
            for p in self.problem.y_encoder.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                # print(this_grad)
                p.grad.copy_(this_grad)
                offset += p.nelement()
        for p in self.MLP.parameters():
            this_grad = grad[offset:offset + p.nelement()].view(p.size())
            p.grad.copy_(this_grad)
            offset += p.nelement()
        optimizer.step()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_train_step(self, train_step):
        self.train_step = train_step

class iMAMLModel(TorchTrainableModel,AdaptiveModel,ProbabilisticModel):
    def __init__(self, problem : SupervisedLearningProblem, params : Union[str,dict]=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
        TorchTrainableModel.__init__(self,problem,params)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = iMAML(problem, params).to(self.device)
        self.fast_model = iMAML(problem, params, True).to(self.device)
        if self.problem.x_encoder is None:
            self.problem.x_encoder = DummyModule()
            self.fast_model.x_encoder = DummyModule()
        if self.problem.y_encoder is None:
            self.problem.y_encoder = DummyModule()
            self.fast_model.y_encoder = DummyModule()
        self.reset()

        #optimizer
        self.optimizer = torch.optim.Adam([{'params': self.problem.x_encoder.parameters()},
                            {'params': self.problem.y_encoder.parameters()},
                            {'params': self.model.MLP.parameters()},],lr = params['outer_lr'])
        self.fast_optimizer = torch.optim.Adam([{'params': self.fast_model.x_encoder.parameters()},
                            {'params': self.fast_model.y_encoder.parameters()},
                            {'params': self.fast_model.MLP.parameters()},],lr = params['inner_lr'])

        self.model.set_optimizer(self.optimizer)
        self.model.set_train_step(self.train_step)

    def train_epoch(self, dataloader, epoch, writer):
        """Override TorchTrainableModel"""
        self.model.train()
        self.fast_model.train()
        meta_grad = 0.0
        query_running_loss = 0
        num_data_in_dataset = len(dataloader.dataset)
        self.optimizer.zero_grad()
        with tqdm(dataloader, unit='batch') as tepoch:
            for i, (support_data, query_data) in enumerate(tepoch):
                support_data = tuple([data.float().to(self.device) for data in support_data])
                query_data = tuple([data.float().to(self.device) for data in query_data])
                task_batch = len(support_data[0])
                initial_weights = get_torchmodel_params(self.model.MLP)
                initial_x_encoder_weights = get_torchmodel_params(self.problem.x_encoder)
                initial_y_encoder_weights = get_torchmodel_params(self.problem.y_encoder)

                for j in range(task_batch):
                    set_torchmodel_params(initial_weights.clone(), self.fast_model.MLP)
                    set_torchmodel_params(initial_x_encoder_weights.clone(), self.fast_model.x_encoder)
                    set_torchmodel_params(initial_y_encoder_weights.clone(), self.fast_model.y_encoder)
                    # learn on data 
                    for _ in range(self.params['inner_epochs']):
                        self.fast_optimizer.zero_grad()
                        inner_loss, p = self.fast_model.fast_model_forward_loss(support_data[0][j], support_data[1][j])
                        inner_loss.backward()
                        self.fast_optimizer.step()

                    # regularization
                    self.fast_optimizer.zero_grad()
                    regu_loss = self.fast_model.regularization_loss(initial_x_encoder_weights, \
                                        initial_y_encoder_weights, initial_weights, self.params['lambda'])
                    regu_loss.backward()
                    self.fast_optimizer.step()

                    # query set loss
                    query_loss, p = self.fast_model.fast_model_forward_loss(query_data[0][j], query_data[1][j])
                    query_running_loss += query_loss.item() 
                    query_grad = torch.autograd.grad(query_loss, list(self.fast_model.x_encoder.parameters()) + \
                        list(self.fast_model.y_encoder.parameters()) + \
                        list(self.fast_model.MLP.parameters()),retain_graph=True)
                    flat_grad = torch.cat([g.contiguous().view(-1) for g in query_grad])
                    task_matrix_evaluator = self.fast_model.matrix_evaluator((support_data[0][j], support_data[1][j]), self.params['lambda'],\
                                                                            self.params['cg_damping'])
                    task_outer_grad = self.fast_model.cg_solve(task_matrix_evaluator, flat_grad, self.params['cg_steps'], x_init=None)

                    meta_grad += (task_outer_grad/task_batch)
        # step outer gradient 
        self.model.outer_step_with_grad(meta_grad, self.optimizer, [initial_x_encoder_weights, initial_y_encoder_weights, \
                initial_weights]) # the last weights argument is just a dummy 
        query_loss = query_running_loss / num_data_in_dataset
        if writer is not None:
            writer.add_scalar("Loss/query", query_loss, epoch)
        print(f"Query loss: {query_loss}")


    def reset(self):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.Tensor([]).float().to(self.device)
        self.support_y = torch.Tensor([]).float().to(self.device)
        self.model.condition_core(self.support_x,self.support_y)
        
    def update(self,x,y):
        """Override AdaptiveScoringModel"""
        self.model.train()
        self.support_x = torch.cat((self.support_x,self.problem.encode_input(totorch(x).unsqueeze(0).float().to(self.device))))
        self.support_y = torch.cat((self.support_y,self.problem.encode_output(totorch(y).unsqueeze(0).float().to(self.device))))
        self.model.condition_core(self.support_x,self.support_y)

    def test_epoch(self, dataloader, epoch, writer):
        """Override to allow gradients."""
        def fs_test_step(supp_data,query_data):
            supp_data = to_device(supp_data,self.device,float)
            query_data = to_device(query_data,self.device,float)
            supp_x,supp_y = supp_data
            query_x,query_y = query_data
            sumloss = None
            for b in range(len(supp_x)):
                loss, p = self.model.forward_fewshot_loss((supp_x[b],supp_y[b]),(query_x[b],query_y[b]))
                if sumloss is None:
                    sumloss = loss
                else:
                    sumloss += loss
            return loss,p #TODO: aggregate predictions?
        return self.fewshot_testing_epoch(dataloader,fs_test_step,writer,epoch)

    def fewshot_testing_epoch(self, dataloader, predict_step,
                            writer=None,epoch=0,classification=False):
        if classification:
            from sklearn.metrics import precision_score, recall_score
        if isinstance(dataloader,DataLoader):
            num_data_in_dataset = len(dataloader.dataset)
        else:
            num_data_in_dataset = len(dataloader)

        val_running_loss = 0
        val_running_corrects = 0
        val_running_MAE = 0
        val_all_preds = []
        val_all_labels = []

        with tqdm(dataloader, unit="batch") as tepoch:
            for i, (support_data,query_data) in enumerate(tepoch):  
                batch_size = query_data[-1][0].shape[0] * query_data[-1][0].shape[1]  #number of tasks * number of query points
                loss, p = predict_step(support_data,query_data)
                p = fromtorch(p)
                labels = fromtorch(query_data[1].float())

                val_running_loss += loss.item() * batch_size
                val_running_MAE += np.mean(np.abs(labels-p))
                if classification:
                    preds = 1.0*(p > 0.5)
                    val_running_corrects += np.sum(preds == labels)
                    val_all_preds += list(preds)
                    val_all_labels += list(labels)

        val_loss = val_running_loss / num_data_in_dataset
        val_MAE = val_running_MAE / num_data_in_dataset

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("MAE/val", val_MAE, epoch)

        if classification:
            val_acc = float(val_running_corrects) / num_data_in_dataset
            val_recall = recall_score(val_all_labels, val_all_preds)
            val_precision = precision_score(val_all_labels, val_all_preds)

            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("Recall/val", val_recall, epoch)
            writer.add_scalar("Precision/val", val_precision, epoch)

            print(f"Val loss: {val_loss}, Val acc: {val_acc}, Val MAE: {val_MAE}, Val recall: {val_recall}, Val precision: {val_precision}")
            return val_loss, val_acc, val_MAE, val_recall, val_precision
        else:
            print(f"Val loss: {val_loss}, Val MAE: {val_MAE}")
            return val_loss, val_MAE


    def __call__(self, x):
        with torch.no_grad():
            x = to_device(totorch(x),self.device,float)
            x,isbatch = tobatch(x)
            res = self.model(x)
            if not isbatch:
                res = frombatch(res)
            if isinstance(res, (list,tuple)):
                return [fromtorch(x) for x in res]
            return fromtorch(res)

        