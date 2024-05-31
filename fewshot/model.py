from abc import ABC,abstractmethod
from typing import Callable,List,Tuple,Union,Optional
import numpy as np
from dataclasses import dataclass
from .data import MultiTaskDataset
import torch.nn as nn
import torch
from enum import Enum

class SupervisedLearningTaskEnum(Enum):
    REGRESSION = 0
    BINARY_CLASSIFICATION = 1
    MULTI_CLASS_CLASSIFICATION = 2

@dataclass
class SupervisedLearningProblem:
    """Defines standard parameters for a supervised learning problem."""
    input_shape : Tuple[int,...]            # shape of the input
    output_shape : Tuple[int,...]           # shape of the output
    input_vars : Optional[List[str]] = None     # names of the input variables somehow broadcastable to the input shape
    output_vars : Optional[List[str]] = None    # names of the output variables somehow broadcastable to the output shape
    x_encoder : Optional[nn.Module] = None  # encoder for the input
    y_encoder : Optional[nn.Module] = None  # encoder for the output
    y_decoder : Optional[nn.Module] = None  # decoder for the output
    encoded_input_shape : Optional[Tuple[int,...]] = None  # shape of the encoded input
    encoded_output_shape : Optional[Tuple[int,...]] = None # shape of the encoded output
    x_encoder_trainable : bool = True       # whether the x_encoder is trainable
    y_encoder_trainable : bool = False      # whether the y_encoder is trainable
    y_decoder_trainable : bool = False      # whether the y_decoder is trainable
    train_on_encoded_x : bool = False       # whether to train on a dataset of encoded inputs.  This can save time if you have a frozen encoder.
    train_on_encoded_y : bool = False       # whether to train on a dataset of encoded outputs.  This can save time if you have a frozen encoder.

    def core_input_shape(self) -> Tuple[int,...]:
        if self.x_encoder is not None:
            assert self.encoded_input_shape is not None
            return self.encoded_input_shape
        else:
            return self.input_shape

    def core_output_shape(self) -> Tuple[int,...]:
        if self.y_encoder is not None:
            assert self.encoded_output_shape is not None
            return self.encoded_output_shape
        else:
            return self.output_shape

    def setup_backbone(self, device : Optional[torch.device]=None):
        """Initializes the backbone on the given device.  If the backbone is
        not trainable, the parameters are frozen.
        """
        if self.x_encoder is not None:
            self.x_encoder.to(device)
            if not self.x_encoder_trainable:
                for param in self.x_encoder.parameters():
                    param.requires_grad = False
        if self.y_encoder is not None:
            self.y_encoder.to(device)
            if not self.y_encoder_trainable:
                for param in self.y_encoder.parameters():
                    param.requires_grad = False
        if self.y_decoder is not None:
            self.y_decoder.to(device)
            if not self.y_decoder_trainable:
                for param in self.y_decoder.parameters():
                    param.requires_grad = False
        
    def encode_input(self,x,in_training=False):
        if self.x_encoder is not None:
            return self.x_encoder(x)
        else:
            return x
    
    def encode_output(self,y):
        if self.y_encoder is not None:
            return self.y_encoder(y)
        else:
            return y
    
    def decode_output(self,y):
        if self.y_decoder is not None:
            return self.y_decoder(y)
        else:
            return y


class TorchModelModule(nn.Module):
    """Base class for a torch module compatible with TorchTrainableModel.
    Unifies the interface to backbones by allowing an implementer to just
    implement the routines for a "core" predictor module.

    Ensures that training will take place with the backbone if the problem
    requests it.

    The core module accepts a (b x n)-D input and produces a (b x m)-D output
    where b is the batch size, n is the input size, and m is the output size.

    This supports normal supervised learning and few-shot learning, and it 
    supports probabilistic and deterministic models.  If the model is few-shot,
    it should implement the `condition_core` method.  If it is probabilistic, 
    the `forward_core` function should return a (means,stds) tuple.  
    """
    def __init__(self,problem : SupervisedLearningProblem):
        super(TorchModelModule,self).__init__()
        self.problem = problem
        input_shape = problem.core_input_shape()
        assert len(input_shape)==1,"Only 1D inputs supported"
        output_shape = problem.core_output_shape()
        assert len(output_shape)==1,"Only 1D outputs supported"
        self.x_encoder = problem.x_encoder if problem.x_encoder_trainable else None
        self.y_encoder = problem.y_encoder if problem.y_encoder_trainable else None

    @abstractmethod
    def forward_core(self,x):
        """Evaluates the core of the model on the given input.  The input
        is already encoded if an encoder is given.  The output is not
        decoded.  If the model is probabilistic, the output should be a
        tuple (means,stds) where means/stds are vectors giving the mean and
        standard deviation of each output.  If the model is not probabilistic,
        the output should be a vector giving the mean of each output.

        x is given with batch size as the leading dimension, i.e., x is a b x n
        tensor.
        """
        raise NotImplementedError()

    @abstractmethod
    def forward_core_loss(self,x,y) -> tuple:
        """Evaluates the core of the model on the given input and produces
        a (loss,prediction) tuple.  x and y are given with batch size as the
        leading dimension, i.e., x has shape b x n and y has shape b x m.
        """
        raise NotImplementedError()
    
    def forward_core_fewshot_loss(self,support_set,query_set) -> tuple:
        """Evaluates the core of the model on the given (support_x,support_y),
        (query_x,query_y) and produces a (loss,prediction) tuple. 
        
        The support and query sets are given with the batch size as the leading
        dimension, i.e., b x k x n support inputs, b x k x m support outputs,
        b x q x n query inputs, b x q x m query outputs.
        """
        support_x, support_y = support_set
        query_x, query_y = query_set
        self.condition_core(support_x,support_y)
        return self.forward_core_loss(query_x,query_y)

    @torch.no_grad()
    def condition_core(self, support_x, support_y):
        """Conditions a few-shot model on the given support set.  """
        raise NotImplementedError("Not a few-shot model")

    def forward(self,x):
        xcore = self.problem.encode_input(x)
        ycore = self.forward_core(xcore)
        if isinstance(ycore,tuple):
            ycore_mean, ycore_std = ycore
            #TODO: the decoder may not properly decode the stds. Would need to decode a perturbed version of the mean or come up with a gradient.
            return self.problem.decode_output(ycore_mean), self.problem.decode_output(ycore_std)
        return self.problem.decode_output(ycore)

    def forward_loss(self,x,y):
        xcore = self.problem.encode_input(x)
        ycore = self.problem.encode_output(y)
        loss,ycore_pred = self.forward_core_loss(xcore,ycore)
        return loss,self.problem.decode_output(ycore_pred)

    def forward_fewshot_loss(self,support_set,query_set):
        support_x, support_y = support_set
        query_x, query_y = query_set
        support_core = (self.problem.encode_input(support_x), self.problem.encode_output(support_y))
        query_core = (self.problem.encode_input(query_x), self.problem.encode_output(query_y))
        loss,ycore_pred = self.forward_core_fewshot_loss(support_core,query_core)
        return loss,self.problem.decode_output(ycore_pred)

    @torch.no_grad()
    def condition(self,support_x,support_y):
        self.condition_core(self.problem.encode_input(support_x),self.problem.encode_output(support_y))


class ProbabilisticModel(ABC):
    """Base class for a probabilistic model that returns a
    tuple (means,stds) where means/stds are vectors giving the 
    mean and standard deviation of each input
    """
    @abstractmethod
    def __call__(self,x) -> Tuple[np.ndarray,np.ndarray]:
        raise NotImplementedError()


class AdaptiveModel(ABC):
    """Base class for an adaptive model that can change
    depending on previously seen (x,y) pairs.
    """
    @abstractmethod
    def update(self,x,y):
        """Adds a new observation to the set of current observations. """
        raise NotImplementedError()
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class TrainableModel(ABC):
    @abstractmethod
    def train(self, train_data : MultiTaskDataset, val_data : Optional[MultiTaskDataset]=None):
        """Trains the model.  If the validation set is given, feedback about
        the training process can be given.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, fn):
        """Saves the model to disk."""
        raise NotImplementedError()
    
    @abstractmethod
    def load(self,fn):
        """Loads the model from disk."""
        raise NotImplementedError()



def adaptive_update(model,x,y):
    """Helper to update an adaptive model.  If the model is not adaptive,
    does nothing."""
    if isinstance(model,AdaptiveModel):
        model.update(x,y)

def adaptive_reset(model):
    """Helper to reset an adaptive model.  If the model is not adaptive,
    does nothing."""
    if isinstance(model,AdaptiveModel):
        model.reset()

def probabilistic_prediction(model,x):
    """Helper to get a probabilistic prediction from a model.  If the model
    is not probabilistic, returns a deterministic prediction (zero standard
    deviation)."""
    if isinstance(model,ProbabilisticModel):
        return model(x)
    else:
        mean = model(x)
        return mean, np.zeros_like(mean)

