import torch
import torch.nn as nn
import os
from .. import utils
from ..model import SupervisedLearningProblem,AdaptiveModel,ProbabilisticModel,TorchModelModule
from ..utils import TorchTrainableModel, weights_init, totorch

class Encoder(nn.Module):
    def __init__(self, sizes, x_dims, y_dims):
        super(Encoder, self).__init__()
        self.linears = utils.MLP(sizes, batchnorm=False)
        self.y_mlp = utils.MLP([y_dims, x_dims], batchnorm=False)

    def forward(self, context_x, context_y=None):
        """
        Encode training elements and aggregate the resulting representations
        into a single vector representation
        Args:
            context_x:  context_set_size x feature_dim
            context_y:  context_set_size x 2 (for classification)
        Returns:
            representations: context_set_size x representation_dim 
        """
        # x = torch.cat((context_x, context_y), dim=-1) # not ideal because of huge difference in dimensionality
        # context_y = context_y.repeat(1,1,context_x.shape[-1]) # broadcasting breaks in case of side info
        context_y = self.y_mlp(context_y)
        x = context_x + context_y
        x = self.linears(x)
        out_shape = list(x.shape)
        out_shape[-1] = -1
        representations = x.view(out_shape)
        return representations

    def init_weights(self):
        self.apply(weights_init)

class Aggregator(nn.Module):
    def __init__(self, sizes):
        super(Aggregator, self).__init__()
        self.linears = utils.MLP(sizes, batchnorm=False)

    def forward(self, representations):
        """
        Aggregate the representations produced by the encoder.
        Args:
            representations: (task_batch) x (context_set_size) x representation_dim
        Returns:
            aggregated_representation: (task_batch) x representation_dim
        """
        representation_dim = representations.shape[-1]
        out_shape = list(representations.shape)
        out_shape[-1] = representation_dim
        mean = representations.mean(dim=-2)
        x = self.linears(mean)
        out = x.view(-1, representation_dim)
        return out

class Decoder(nn.Module):
    def __init__(self, sizes):
        super(Decoder, self).__init__()
        self.linears = utils.MLP(sizes, batchnorm=False, output='Sigmoid')

    def forward(self, query_x, representation):
        """
        General module for input-target pair representation
        Args:
            representation: (task_batch) x representation_dim
            query_x: (task_batch) x (query_set_size) x feature_dim
        Returns:
            out: mean and sd of the distribution
            (task_batch) x (query_set_size) x out_feature_dim
        """
        out_shape = list(query_x.shape)[:-1]+[-1]
        representation_dim = representation.shape[-1]
        input_dim = query_x.shape[-2]
        feature_dim = query_x.shape[-1]

        query_x = query_x.view(-1, feature_dim)

        representation = representation.unsqueeze(-2).repeat(1, input_dim, 1)
        representation = representation.view(-1, representation_dim)

        decoder_input = torch.cat((representation, query_x), dim=-1)
        x = decoder_input.view(out_shape)
        x = self.linears(x)
        out = x.view(out_shape)
        mu, log_sigma = torch.split(out, 1, dim=-1)
        mu, sigma = 1e-3 + mu, 1e-3 + torch.exp(log_sigma)
        return mu, sigma

    def init_weights(self):
        self.apply(weights_init)
    
class CNP(nn.Module):
    """
    Takes a datapoint of context_x, context_y, target_x and predicts target_y. 
    Returns:
        mu: batch_size x 1
        sigma: batch_size x 1
    """

    def __init__(self, x_dims, y_dims, params):
        super(CNP, self).__init__()
        self.params = params

        # Loading sub module architectures
        encoder_sizes = [x_dims] + params['encoder_hlayers'] + [params['representation_size']]
        aggregator_sizes = [encoder_sizes[-1]] + params['aggregator_hlayers'] + [params['representation_size']]
        decoder_sizes = [aggregator_sizes[-1] + x_dims] + params['decoder_hlayers'] + [params['d_out']]

        self.encoder = Encoder(encoder_sizes, x_dims, y_dims) #r
        self.aggregator = Aggregator(aggregator_sizes) #w
        self.decoder = Decoder(decoder_sizes)#o
    
    def forward(self, support_x, support_y, query_x):
        #TODO: aggregate the representation incrementally for faster updates
        if len(support_x) == 0:
            representation = torch.zeros(1, self.params['representation_size']).to(support_x.device)
        else:
            representations = self.encoder(support_x, support_y)
            representation = self.aggregator(representations)
        mu, sigma = self.decoder(query_x, representation)
        return mu, sigma


class ConditionalNP(TorchModelModule):
    def __init__(self, problem : SupervisedLearningProblem, params):
        super(ConditionalNP, self).__init__(problem)

        self.params = params
        x_dims = problem.core_input_shape()[0]
        y_dims = problem.core_output_shape()[0]
        self.model = CNP(x_dims, y_dims, self.params)
        self.support_x = None
        self.support_y = None

    def forward_core(self, x):
        assert self.support_x is not None and self.support_y is not None, "CNP must be conditioned before forward_core is called."
        mu, sigma = self.model(self.support_x, self.support_y, x)
        return mu, sigma

    def forward_core_fewshot_loss(self, support_data, query_data):
        support_x, support_y = support_data
        query_x, query_y = query_data
   
        mu_preds, sigma_preds = self.model(support_x, support_y, query_x)

        dist = torch.distributions.normal.Normal(loc=mu_preds, scale=sigma_preds)
        log_prob = dist.log_prob(query_y)
        loss = -log_prob.mean()
        loss_sum = -log_prob.sum()
        #return loss, loss_sum, mu_preds, sigma_preds
        return loss, mu_preds

    @torch.no_grad()
    def condition_core(self, support_x, support_y):
        self.support_x = support_x
        self.support_y = support_y


class CNPModel(TorchTrainableModel,AdaptiveModel,ProbabilisticModel):
    def __init__(self, problem : SupervisedLearningProblem, params=None):
        if params is None:
            params = os.path.join(os.path.split(__file__)[0],'config.yaml')
        assert 'k_shot' in params, "k_shot must be specified in the config file since only few-shot learning is supported."
        TorchTrainableModel.__init__(self, problem, params)
        self.model = ConditionalNP(self.problem, params).to(self.device)
        self.reset()
    
    def train_step(self,support_data,query_data):
        loss, mu_preds = self.model.forward_fewshot_loss(support_data, query_data)
        return loss,mu_preds
        
    def test_step(self,support_data,query_data):
        loss, mu_preds = self.model.forward_fewshot_loss(support_data, query_data)
        return loss,mu_preds
        
    def reset(self):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.Tensor([]).float().to(self.device)
        self.support_y = torch.Tensor([]).float().to(self.device)
        self.model.condition_core(self.support_x,self.support_y)
        
    def update(self,x,y):
        """Override AdaptiveScoringModel"""
        self.support_x = torch.cat((self.support_x,self.problem.encode_input(totorch(x).float().to(self.device)).unsqueeze(0)),0)
        self.support_y = torch.cat((self.support_y,self.problem.encode_output(totorch(y).float().to(self.device)).unsqueeze(0)),0)
        self.model.condition_core(self.support_x,self.support_y)
