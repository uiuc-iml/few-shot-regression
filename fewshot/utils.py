from abc import abstractmethod
import os
import random
from typing import Callable,Optional,Union,Tuple,List
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import _utils
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import yaml
from tqdm import tqdm
from .model import SupervisedLearningProblem,TrainableModel,TorchModelModule
from .data import StandardMultiTaskDataset,MultiTaskDataset,FlatDataset,FewShotDataset
from dataclasses import dataclass,asdict,replace


def set_seed(seed, verbose=True):
    """Sets a common seed amongst random, numpy and torch"""
    if(seed!=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if(verbose): print("[INFO] Setting SEED: " + str(seed))
    else:
        if(verbose): print("[INFO] Setting SEED: None")

def to_device(x,device,type=None):
    if isinstance(x,tuple):
        return tuple([to_device(data,device,type) for data in x])
    elif isinstance(x,list):
        return [to_device(data,device,type) for data in x]
    if type is None:
        return x.to(device)
    elif type is float:
        return x.float().to(device)
    elif type is int:
        return x.int().to(device)
    else:
        raise ValueError("Unknown type")

def load_config(config_fn : str, variant : Union[str,List[str]] = None):
    """Loads a configuration YAML file.
    
    If variant is provided, it can be a str or list of str.  The "variant"
    key in the config dict will be used to update the config dict.
    """
    with open(os.path.join(config_fn)) as file:
        config = yaml.safe_load(file)

    if variant is not None:
        if 'variants' in config:
            if isinstance(variant,str):
                variant = [variant]
            for v in variant:
                if v in config['variants']:
                    variant_config = config['variants'][v]
                    config.update(variant_config)
                else:
                    print(f"Variant '{v}' not found in config file, possible options:")
                    for v2 in config['variants']:
                        print(" ",v2)
                    raise ValueError(f"Variant '{v}' not found in config file")
            config.pop('variants')
        else:
            raise ValueError("No variants found in config file")
    return config


def weights_init(m):
    """Initializes a layer of a neural network with xavier initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    return


class MLP(nn.Module):
    """A basic customizable Multi-Layer-Perceptron."""
    def __init__(self,layer_sizes,nonlinear='ReLU',batchnorm=False,output=None,dropout=0.0):
        super(MLP, self).__init__()
        self.layers = []
        for i in range(len(layer_sizes)-1):
            args = [nn.Linear(layer_sizes[i], layer_sizes[i+1])]
            # args.append(nn.Dropout(p=dropout))
            if batchnorm:
                args.append(nn.BatchNorm1d(layer_sizes[i+1]))
            if i+2 < len(layer_sizes):
                if nonlinear=='LeakyReLU':
                    args.append(nn.LeakyReLU(negative_slope=0.0))
                elif nonlinear=='ReLU':
                    args.append(nn.ReLU())
                elif nonlinear=='Sigmoid':
                    args.append(nn.Sigmoid())
                elif nonlinear=='Tanh':
                    args.append(nn.Tanh())
                else:
                    assert not isinstance(nonlinear,str),"Must give a nonlinear transform layer"
                    args.append(nonlinear())
            else:
                if output == 'Sigmoid':
                    args.append(nn.Sigmoid())
            self.layers.append(nn.Sequential(*args))
        #need to do this so that PyTorch can find the layer
        for i,layer in enumerate(self.layers):
            setattr(self,'layer{}'.format(i),layer)

        self.init_weights()

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        self.apply(weights_init)


def standard_training_epoch(dataloader, optimize_step:Callable,
                            writer=None,epoch=0,classification=False):
    """Run an epoch of standard training, with proper printouts."""
    if classification:
        from sklearn.metrics import precision_score, recall_score

    if isinstance(dataloader,DataLoader):
        num_data_in_dataset = len(dataloader.dataset)
    else:
        num_data_in_dataset = len(dataloader)
    train_running_loss = 0
    train_running_corrects = 0
    train_running_MAE = 0
    train_all_preds = []
    train_all_labels = []

    with tqdm(dataloader, unit="batch") as tepoch:
        for i, (x,y) in enumerate(tepoch):
            batch_size = len(x)
            loss, p = optimize_step(x,y)
            p = fromtorch(p)

            labels = fromtorch(y.float())

            train_running_loss += loss.item() * batch_size
            train_running_MAE += np.mean(np.abs(labels-p))
            if classification:
                preds = fromtorch(p.ge(0.5).float())
                train_running_corrects += np.sum(preds == labels)
                train_all_preds += list(preds)
                train_all_labels += list(labels)
                recall = recall_score(labels, preds)
                precision = precision_score(labels, preds)

        tepoch.set_postfix(Epoch=epoch,loss=train_running_loss/((i+1)*batch_size))

    epoch_loss = train_running_loss / num_data_in_dataset
    epoch_MAE = train_running_MAE / num_data_in_dataset
    if writer is not None:
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("MAE/train", epoch_MAE, epoch)
    
    if classification:
        epoch_acc = float(train_running_corrects) / num_data_in_dataset
        epoch_recall = recall_score(train_all_labels, train_all_preds)
        epoch_precision = precision_score(train_all_labels, train_all_preds)
        print(f"Epoch loss: {epoch_loss}, Epoch acc: {epoch_acc}, Epoch MAE: {epoch_MAE}, Epoch recall: {epoch_recall}, Epoch precision: {epoch_precision}")
    else:
        print(f"Epoch loss: {epoch_loss}, Epoch MAE: {epoch_MAE}")


def standard_testing_epoch(dataloader, predict_step:Callable,
                            writer=None,epoch=0,classification=False):
    """Run an epoch of standard testing, with proper printouts."""
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
        for x,y in tepoch:
            batch_size = len(x)
            with torch.no_grad():
                loss, p = predict_step(x,y)
            p = fromtorch(p)
            labels = fromtorch(y.float())

            val_running_loss += loss.item() * batch_size
            val_running_MAE += np.mean(np.abs(labels-p))
            if classification:
                #download to cpu
                preds = fromtorch(p.ge(0.5).float())
                val_running_corrects += np.sum(preds == labels)
                val_all_preds += list(preds)
                val_all_labels += list(labels)

    val_loss = val_running_loss / num_data_in_dataset
    val_MAE = val_running_MAE / num_data_in_dataset
    if writer is not None:
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("MAE/val", val_MAE, epoch)

    if classification:
        val_acc = float(val_running_corrects) / num_data_in_dataset
        val_recall = recall_score(val_all_labels, val_all_preds)
        val_precision = precision_score(val_all_labels, val_all_preds)
        if writer is not None:
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("Recall/val", val_recall, epoch)
            writer.add_scalar("Precision/val", val_precision, epoch)

        print(f"Val loss: {val_loss}, Val acc: {val_acc}, Val MAE: {val_MAE}, Val recall: {val_recall}, Val precision: {val_precision}")
        return val_loss, val_acc, val_MAE, val_recall, val_precision
    else:
        print(f"Val loss: {val_loss}, Val MAE: {val_MAE}")
        return val_loss, val_MAE


def fewshot_training_epoch(dataloader, optimize_step:Callable,
                            writer=None,epoch=0,classification=False, return_loss = False):
    """Run an epoch of standard few-shot training, with proper printouts."""
    if classification:
        from sklearn.metrics import precision_score, recall_score
    if isinstance(dataloader,DataLoader):
        assert isinstance(dataloader.dataset, FewShotDataset), "Few-shot training requires a few-shot dataset to be configured"
        num_data_in_dataset = len(dataloader.dataset)
    else:
        num_data_in_dataset = len(dataloader)
    train_running_loss = 0
    train_running_corrects = 0
    train_running_MAE = 0
    train_all_preds = []
    train_all_labels = []

    with tqdm(dataloader, unit="batch") as tepoch:
        for i, (support_data, query_data) in enumerate(tepoch):
            batch_size = query_data[-1][0].shape[0] * query_data[-1][0].shape[1]  #number of tasks * number of query points
            loss, p = optimize_step(support_data,query_data)
            p = fromtorch(p)
            out = query_data[1]
            labels = fromtorch(out.float()) 
            train_running_loss += loss.item() * batch_size  #assumes loss is MSE?
            train_running_MAE += np.mean(np.abs(labels-p))
            if classification:
                preds = fromtorch(p.ge(0.5).float())
                train_running_corrects += np.sum(preds == labels)
                train_all_preds += list(preds)
                train_all_labels += list(labels)

                recall = recall_score(labels, preds)
                precision = precision_score(labels, preds)

        tepoch.set_postfix(Epoch=epoch,loss=train_running_loss/((i+1)*batch_size))

    train_loss = train_running_loss / num_data_in_dataset
    train_MAE = train_running_MAE / num_data_in_dataset
    
    if writer is not None:
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("MAE/train", train_MAE, epoch)
        #writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], batch_size)

    epoch_loss = train_running_loss / num_data_in_dataset
    epoch_MAE = train_running_MAE / num_data_in_dataset
    if classification:
        epoch_acc = float(train_running_corrects) / num_data_in_dataset
        epoch_recall = recall_score(train_all_labels, train_all_preds)
        epoch_precision = precision_score(train_all_labels, train_all_preds)
        if writer is not None:
            writer.add_scalar("Acc/train", epoch_acc, epoch)
            writer.add_scalar("Recall/train", epoch_recall, epoch)
            writer.add_scalar("Precision/train", epoch_precision, epoch)

        print(f"Epoch loss: {epoch_loss}, Epoch acc: {epoch_acc}, Epoch MAE: {epoch_MAE}, Epoch recall: {epoch_recall}, Epoch precision: {epoch_precision}")
    else:
        print(f"Epoch loss: {epoch_loss}, Epoch MAE: {epoch_MAE}")

    if return_loss:
        return train_loss

def fewshot_testing_epoch(dataloader, predict_step:Callable,
                            writer=None,epoch=0,classification=False):
    """Run an epoch of standard testing, with proper printouts."""
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
            with torch.no_grad():
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


@dataclass
class TorchTrainingParams:
    """Standard parameters for training a TorchTrainableModel.  These will 
    typically be loaded from a YAML file into a dictionary, but can also be
    used directly as an argument to TorchTrainableModel.

    Learning rate decay is applied every `lr_step` epochs, with the learning rate
    multiplied by `lr_decay`.

    Early stopping is applied with `patience` epochs of no improvement.

    Attributes:
    - epochs: number of epochs to train
    - device: default 'cuda:0' if available, else 'cpu'
    - batch_size: batch size for training
    - num_workers: number of workers for data loading. default 0
    - seed: RNG seed. default None
    - optimizer: default 'Adam'
    - lr: learning rate
    - lr_decay: learning rate decay factor, default None
    - lr_step: learning rate decay step, default None
    - log_dir: log directory for the writer object (default 'training_logs/')
    - checkpoint_dir: model checkpoint directory
    - patience: early stopping patience. default 100
    - k_shot: either an int or list of ints if few-shot training is used.
        defaults to None, which uses standard training.
    - num_draws_per_task: # of draws used per task. default 10
    - resample_per_epoch: if you would like to resample the few-shot training
        set each epoch, set this to True.  Default False.
    - query_set_size: if not None, few-shot training will use this number
        of query points.  Default None.
    - task_batch_size: if few-shot training is used, the number of tasks in a
        batch.  If k_shot is variable or query sets have variable size, this
        must be 1.  Default 1.
    - train_x_encoder: whether to train the input encoder. default True
    - train_y_encoder: whether to train the output encoder. default True
    - x_encoder_weight_fn: file for pretrained input encoder
    - y_encoder_weight_fn: file for pretrained output encoder
    """
    epochs: int = 50
    device : Optional[str] = None
    batch_size: int = 128
    num_workers: int = 0
    seed : Optional[int] = None
    optimizer: str = 'Adam'
    lr: float = 0.1
    lr_decay: float = 0.7
    lr_step: int = 20
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    val_freq: int = 1
    patience: int = 100
    k_shot: Optional[Union[int,List[int]]] = None
    num_draws_per_task: int = 10
    resample_per_epoch : bool = False
    query_set_size: Optional[int] = None
    task_batch_size: int = 1
    train_x_encoder : bool = True
    train_y_encoder : bool = True
    x_encoder_weight_fn : Optional[str]=None
    y_encoder_weight_fn : Optional[str]=None


class TorchTrainableModel(TrainableModel):
    """A "standard" supervised learning model in PyTorch.  A base class for all
    PyTorch models, which can accommodate few-shot and standard regresion training.
    
    Standard usage:
    - Subclass and set self.model to a TorchModelModule
    - Optionally override train_step and test_step for custom training/testing
    - Optionally override train_epoch and test_epoch for custom training/testing
    - Optionally override __call__ for custom evaluation

    Users will run something like this::
    
        #training
        ##setup problem and params
        problem = SupervisedLearningProblem()
        params = TorchTrainingParams()
        ##setup training and validation data
        train_data = TRAIN_DATA (either a MultiTaskDataset or a standard dataloader-compatible structure)
        val_data = VAL_DATA (either a MultiTaskDataset or a standard dataloader-compatible structure)
        ##create, train, and save model
        model = MyTorchModel(problem, params)
        model.train(train_data, val_data)
        model.save('my_model.tar')

        #testing
        model.load('my_model.tar')
        x = TEST_DATUM
        ypred = model(x)

    Standard parameters are given by the attributes in TorchTrainingParams.

    Training uses multiple epochs of SGD training. If you want to use Tensorboard to
    visualize training, pass in a SummaryWriter to train(), and its results will be
    written to the folder given by `params['log_dir']`.
    """
    def __init__(self,
                 problem: SupervisedLearningProblem,
                 params : Union[str,dict,TorchTrainingParams]):
        # Load config
        if isinstance(params,str):
            self.params_file = params
            params = load_config(params)
        elif isinstance(params,TorchTrainingParams):
            params = asdict(params)
            self.params_file = None
        else:
            self.params_file = None
        self.problem = problem
        self.params = params
        self.model = None                   # type: TorchModelModule
        if self.params.get('device', None):
            self.device = torch.device(self.params['device'])
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load backbone to device
        self.problem.setup_backbone()
        #store backbone as attributes so they are saved and loaded as part of state_dict()
        self.x_encoder = self.problem.x_encoder
        self.y_encoder = self.problem.y_encoder

        # load pretrained observation_action_encoder
        if self.params.get('x_encoder', None):
            fn = self.params['x_encoder']
            checkpoint = torch.load(fn, map_location=self.device)
            self.problem.x_encoder.load_state_dict(checkpoint['state'])
        
        if self.params.get('y_encoder', None):
            fn = self.params['y_encoder']
            checkpoint = torch.load(fn, map_location=self.device)
            self.problem.y_encoder.load_state_dict(checkpoint['state'])
        
        if self.params.get('seed',None) != None:
            set_seed(self.params['seed'])
        
        self.optimizer = None

    def fewshot_training(self):
        return self.params.get('k_shot',None)
    
    def train_step(self,x,y) -> tuple:
        """Subclass may override me.  Returns (loss,core_predictions) tuple
        for a batch of data. 
        
        In the normal case, x is an input batch and y is an output batch.

        In the few-shot case, x is the (support set x, support set y) tuple and
        y is the (query set x, query set y) tuple.  Both are given with the batch
        size as the leading dimension.
        """
        if self.fewshot_training():
            return self.model.forward_fewshot_loss(x,y)
        else:
            return self.model.forward_loss(x,y)

    def test_step(self,x,y):
        """Subclass may override me.  Returns (loss,core_predictions) tuple
        for a batch of data.  Default is to call model.forward_loss.

        In the normal case, x is an input batch and y is an output batch.
        
        In the few-shot case, x is the (support set x, support set y) tuple and
        y is the (query set x, query set y) tuple.  Both are given with the batch
        size as the leading dimension.
        """
        if self.fewshot_training():
            return self.model.forward_fewshot_loss(x,y)
        else:
            return self.model.forward_loss(x,y)

    def train_epoch(self, dataloader, epoch, writer):
        """Subclass may override me."""
        self.model.train()
        def optimize_step(x,y):
            x = to_device(x,self.device,float)
            y = to_device(y,self.device,float)
            self.optimizer.zero_grad()
            loss, p = self.train_step(x,y)
            loss.backward()
            self.optimizer.step()
            return loss, p
        def fs_optimize_step(supp_data,query_data):
            supp_data = to_device(supp_data,self.device,float)
            query_data = to_device(query_data,self.device,float)
            supp_x,supp_y = supp_data
            query_x,query_y = query_data
            self.optimizer.zero_grad()
            preds = []
            sumloss = None
            for b in range(len(supp_x)):
                loss, p = self.train_step((supp_x[b],supp_y[b]),(query_x[b],query_y[b]))
                if sumloss is None:
                    sumloss = loss
                else:
                    sumloss += loss
                preds.append(p)
            sumloss.backward()
            self.optimizer.step()
            return loss, torch.stack(preds,dim=0)  #aggregate predictions
        if self.fewshot_training():
            fewshot_training_epoch(dataloader, fs_optimize_step, writer,epoch)
        else:
            standard_training_epoch(dataloader, optimize_step, writer, epoch)

    @torch.no_grad()
    def test_epoch(self, dataloader, epoch, writer):
        """Subclass may override me."""
        self.model.eval()
        def test_step(x,y):
            x = to_device(x,self.device,float)
            y = to_device(y,self.device,float)
            loss, p = self.test_step(x,y)
            return loss,p
        def fs_test_step(supp_data,query_data):
            supp_data = to_device(supp_data,self.device,float)
            query_data = to_device(query_data,self.device,float)
            supp_x,supp_y = supp_data
            query_x,query_y = query_data
            sumloss = None
            for b in range(len(supp_x)):
                loss, p = self.test_step((supp_x[b],supp_y[b]),(query_x[b],query_y[b]))
                if sumloss is None:
                    sumloss = loss
                else:
                    sumloss += loss
            return loss,p #TODO: aggregate predictions?
        
        if self.params.get('k_shot',None):
            return fewshot_testing_epoch(dataloader,fs_test_step,writer,epoch)
        else:
            return standard_testing_epoch(dataloader,test_step,writer,epoch)

    def train(self, train_data : MultiTaskDataset, val_data : Optional[MultiTaskDataset]=None, writer : Optional[SummaryWriter] = None):
        """Performs training of the model according to the given parameters. 
        For implementers: if you have special training procedures, it is
        recommended not to override this, but to instead override
        train_epoch / test_epoch.
        
        Overrides TrainableModel.
        """
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

        st = time.time()
        support_set_size = self.params.get('k_shot',None)
        resample_per_epoch = self.params.get('resample_per_epoch',False)
        batch_size = self.params.get('batch_size',None)
        num_workers = self.params.get('num_workers',0)
        if support_set_size is None:
            train_dataset = FlatDataset(train_data)
            print("Using a flattened dataset of length",len(train_dataset))
            if batch_size is None:
                batch_size = len(train_dataset)
            #Note: resample_per_epoch doesn't do anything: FlatDataset naturally resamples from the dataset
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)#, pin_memory=True)
            
            # set batch_size  and num_workers to 1 and keep them fixed for consistency across experiments
            if val_data is not None:
                st = time.time()
                val_dataset = FlatDataset(val_data)
                if batch_size is not None:
                    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)#, pin_memory=True)
                else:
                    val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=num_workers)#, pin_memory=True)
        else:
            #few-shot training
            query_set_size = self.params.get('query_set_size',None)
            num_draws_per_task = self.params.get('num_draws_per_task',10)
            task_batch_size = self.params.get('task_batch_size',1)
            if isinstance(support_set_size,list):
                if task_batch_size > 1:
                    raise ValueError("task_batch_size must be 1 for variable support set size, got {}".format(task_batch_size))\
        
            train_task_dataset = FewShotDataset(train_data,support_set_size,query_set_size,num_draws_per_task)
            print("Size of few-shot dataset: ",len(train_task_dataset))
            print("Average support set size",np.mean([len(s[0][0]) for s in train_task_dataset]))
            print("Average query set size",np.mean([len(s[1][0]) for s in train_task_dataset]))
            if resample_per_epoch:
                train_loader = lambda : DataLoader(FewShotDataset(train_data,support_set_size,query_set_size,num_draws_per_task), batch_size=task_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)#, pin_memory=True)
            else:
                train_loader = DataLoader(train_task_dataset, batch_size=task_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)#, pin_memory=True)
                if time.time()-st > 1.0:
                    print ('time to construct train data / dataloader: ', time.time()-st, len(train_loader.dataset))

            if val_data is not None:
                # For validation, we only look at highest-shot k-shot for clean validation loss curve
                t0 = time.time()
                if isinstance(support_set_size, list):
                    support_set_size=max(support_set_size)
                val_task_dataset = FewShotDataset(val_data,support_set_size)
                val_loader = DataLoader(val_task_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate)#, pin_memory=True)
                if time.time()-t0 > 1.0:
                    print ('time to construct val data / dataloader: ', time.time()-t0, len(val_loader.dataset))

        self.train_loop(train_loader, val_loader, writer)

        if created_writer:
            writer.flush()
            writer.close()

    def train_loop(self, train_loader : Union[DataLoader,Callable], val_loader : Optional[DataLoader], writer : Optional[SummaryWriter]):
        # initialize the early_stopping object
        patience = self.params.get('patience',100)
        if patience < self.params['epochs']:
            self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        else:
            self.early_stopping = None
        if callable(train_loader):
            train_loader_func = train_loader
        else:
            train_loader_func = lambda: train_loader

        LEARNING_RATE_CLIP = 1e-5

        #begin training
        for epoch in range(self.params['epochs']):
            print("Epoch",epoch)
            if self.params.get('lr_decay',0):
                for g in self.optimizer.param_groups:
                    g['lr'] = max(self.params['lr'] * (self.params['lr_decay'] ** (epoch // self.params['lr_step'])), LEARNING_RATE_CLIP)

            train_loader = train_loader_func()  #generate a train loader -- can be used for data augmentation
            self.train_epoch(train_loader, epoch, writer)

            res = self.test_epoch(val_loader, epoch, writer)
            valid_loss = res[0]

            if self.early_stopping is not None:
                self.early_stopping(valid_loss, self)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
        
        if hasattr(self,'reset'):
            #during training, some support sets may have been set. These need to be cleared
            self.reset()


    def load(self, fn=None):
        if fn is None:
            if 'checkpoint_dir' not in self.params:
                raise ValueError("No file or checkpoint directory specified")
            folder_path = self.params['checkpoint_dir']
            fn = os.path.join(folder_path, 'model.tar')
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint)
        print('Reloaded trained model from {}'.format(fn))
        self.model.eval()

    def save(self, fn=None):
        if fn is None:
            if 'checkpoint_dir' not in self.params:
                raise ValueError("No file or checkpoint directory specified")
            folder_path = self.params['checkpoint_dir']
            fn = os.path.join(folder_path, 'model.tar')
        print(f'saving to {fn}')
        torch.save(self.model.state_dict(),fn)

    def __call__(self,x):
        """Evaluate on one or more inputs.  x can be an array-like object or a
        sequence of array-like objects. 
        
        x is assumed to either be a batch or a single input. Single inputs are
        detected if any item in x is 1D.  Note that this detection method may be
        imperfect if your inputs are, say, images.  In this case, you need to
        ensure that your inputs are batched properly.

        Subclass may override me.
        """
        with torch.no_grad():
            x = to_device(totorch(x),self.device,float)
            x,isbatch = tobatch(x)
            res = self.model(x)
            if not isbatch:
                res = frombatch(res)
            if isinstance(res, (list,tuple)):
                return [fromtorch(x) for x in res]
            return fromtorch(res)
    

class TwoStageTorchTrainableModel(TorchTrainableModel):
    """A base class for fewshot PyTorch models with two stages, a zero-shot trainig stage and an adaptive stage.
    
    Standard usage:
    - Override zero_shot_train
    - Override adaptive_train
    """
    def __init__(self,
                problem: SupervisedLearningProblem,
                params : Union[str,dict,TorchTrainingParams]):
        super().__init__(problem, params)

    def zero_shot_train(self, train_loader, val_loader, write = None):
        raise NotImplementedError()

    def adaptive_train(self, train_loader, val_loader, write = None):
        raise NotImplementedError()

    def train(self, train_data : MultiTaskDataset, val_data : Optional[MultiTaskDataset]=None, writer : Optional[SummaryWriter] = None):
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
        val_loader = None
        if val_data is not None:
            st = time.time()
            zero_shot_val_dataset = FlatDataset(val_data)
            if  zero_shot_batch_size is not None:
                zero_shot_val_loader = DataLoader(dataset= zero_shot_val_dataset, batch_size= zero_shot_batch_size, shuffle=False, num_workers= zero_shot_num_workers)#, pin_memory=True)
            else:
                zero_shot_val_loader = DataLoader(dataset= zero_shot_val_dataset, batch_size=len( zero_shot_val_dataset), shuffle=False, num_workers= zero_shot_num_workers)#, pin_memory=True)
        
        self.zero_shot_train(zero_shot_train_loader, zero_shot_val_loader, writer)

        #few-shot training
        support_set_size = self.params.get('k_shot',None)    
        query_set_size = self.params.get('query_set_size',None)
        num_draws_per_task = self.params.get('num_draws_per_task',10)
        task_batch_size = self.params.get('task_batch_size',1)
        if isinstance(support_set_size,list):
            if task_batch_size > 1:
                raise ValueError("task_batch_size must be 1 for variable support set size, got {}".format(task_batch_size))
        train_task_dataset = FewShotDataset(train_data,support_set_size,query_set_size,num_draws_per_task)
        print("Size of few-shot dataset: ",len(train_task_dataset))
        print("Average support set size",np.mean([len(s[0][0]) for s in train_task_dataset]))
        print("Average query set size",np.mean([len(s[1][0]) for s in train_task_dataset]))
        train_loader = DataLoader(train_task_dataset, batch_size=task_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)#, pin_memory=True)

        if val_data is not None:
            # For validation, we only look at highest-shot k-shot for clean validation loss curve
            t0 = time.time()
            if isinstance(support_set_size, list):
                support_set_size=max(support_set_size)
            val_task_dataset = FewShotDataset(val_data,support_set_size)
            val_loader = DataLoader(val_task_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate)#, pin_memory=True)
            if time.time()-t0 > 1.0:
                print ('time to construct val data / dataloader: ', time.time()-t0, len(val_loader.dataset))

        self.adaptive_zero_shot_train(train_loader, val_loader, writer)

        self.save()
        if hasattr(self,'reset'):
            #during training, some support sets may have been set. These need to be cleared
            self.reset()

        if created_writer:
            writer.flush()
            writer.close()

def totorch(data):
    """Converts tensors, numpy arrays, and lists of such items to torch Tensors.
    The resulting tensors will be of float datatype."""
    if isinstance(data,list):
        if len(data) == 0:
            return torch.Tensor(data)
        return _utils.collate.default_collate(data).float()
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return _utils.collate.default_convert(data).float()
    
    print("Warning: totorch is operating on an object of type",data.__class__.__name__)
    print(data)
    return _utils.collate.default_convert(data).float()

def fromtorch(data):
    """Converts a torch tensor or list/tuple of torch tensors to numpy
    arrays of the same shape."""
    if isinstance(data,torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data,list):
        return [fromtorch(d) for d in data]
    elif isinstance(data,tuple):
        return tuple([fromtorch(d) for d in data])
    return data

def tolist(data) -> list:
    """Converts a torch tensor, numpy array or list/tuple of such objects to lists
    of the same shape."""
    if isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (list, tuple)):
        return [tolist(d) for d in data]
    return data

def tobatch(data,force=False) -> Tuple[torch.Tensor,bool]:
    """Returns tuple (batched_data,isbatch) to be ready to fed into torch
    Modules.

    Detects single inputs by checking whether the data data has a 1D element,
    and unsqueezes if so.

    If your data is a 2D single item, then you need to set force=True to ensure
    that it is batched properly.  
    """
    if isinstance(data, torch.Tensor):
        if len(data.shape) == 1 or force:
            return data.unsqueeze(0), False
        return data, True
    elif isinstance(data, (list, tuple)):
        isbatch = []
        batched_data = []
        for d in data:
            dbatch, disbatch = tobatch(d, force)
            batched_data.append(dbatch)
            isbatch.append(disbatch)
        if all(isbatch):
            return batched_data, True
        elif any(isbatch):
            #need to unsqueeze anything that isn't a batch
            for i,b in enumerate(isbatch):
                if b:
                    batched_data[i] = batched_data[i].unsqueeze(0)
        return batched_data, False
    raise ValueError("Invalid type of data sent to tobatch, must be a Tensor or sequence of Tensors")

def frombatch(data : Union[torch.Tensor,Tuple]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.squeeze(0)
    elif isinstance(data, (list, tuple)):
        return [frombatch(d) for d in data]
    raise ValueError("Invalid type of data sent to frombatch, must be a Tensor or sequence of Tensors")

def collate(data):
    return _utils.collate.default_collate(data)
    
class EarlyStopping:
    '''This class is copied from github repo: https://github.com/Bjarten/early-stopping-pytorch'''
    
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model = None

    def __call__(self, val_loss, model:TorchTrainableModel):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model:TorchTrainableModel):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save()
        self.val_loss_min = val_loss
