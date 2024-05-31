import numpy as np
import pandas as pd
import torch
import scipy
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, List
from .benchmark import LearningBenchmark
from ..data import TaskDataset, MultiTaskDataset, uniform_multi_task_sampler, add_dataset_noise
from ..model import SupervisedLearningProblem
from .. import utils

class ToyGaussianFunction:
    """Returns the height of a gaussian centerd at mu with standard deviation std at x.

    Can be run conditioned on a (mu,sd) tuple or not

    May also include derivative or displacement side information.
    """
    def __init__(self, mu : float, std : float, observe_derivative = False, observe_displacement = False):
        self.mu = mu
        self.std = std
        self.pdf = scipy.stats.norm(mu, std)
        self.observe_derivative = observe_derivative
        self.observe_displacement = observe_displacement
        self._cached_mu = None
        self._cached_std = None
        self._cached_pdf = None
    
    def __call__(self,x, mu_std = None):
        if mu_std is None:
            pdf = self.pdf
        else:
            mu,std = mu_std
            if mu != self._cached_mu or std != self._cached_std:
                self._cached_pdf = scipy.stats.norm(mu, std)
                self._cached_mu = mu
                self._cached_std = std
            pdf = self._cached_pdf
        x = x[0]
        outputs = [pdf.pdf(x)]
        h = 0.0001
        if self.observe_derivative:
            outputs.append((pdf.pdf(x + h) - pdf.pdf(x))/h )
        if self.observe_displacement:
            outputs.append(x - self.mu)
        return np.array(outputs)


class ToyGaussianBenchmark(LearningBenchmark):
    """Returns a dataset of gaussian functions"""
    def __init__(self, variant : Union[str,List[str]] = None):
        import os
        config_file = os.path.join(os.path.dirname(__file__), 'toy_gaussian_config.yaml')
        self.params = utils.load_config(config_file, variant)

    def problem(self):
        sideinfo = self.params['sideinfo']
        func = ToyGaussianFunction(0,1.0,sideinfo=='derivative',sideinfo=='displacement')
        n_out = len(func([0.0]))
        output_vars = ['height']
        if sideinfo:
            output_vars.append(sideinfo)
        return SupervisedLearningProblem(input_shape=(1,),output_shape=(n_out,),output_vars=output_vars)
    
    def dataset(self, val_split = 0.2) -> Tuple[MultiTaskDataset, MultiTaskDataset, MultiTaskDataset]:
        params = self.params
        num_tasks = params['num_tasks']
        num_samples_per_task = params['num_samples_per_task']
        x_range = params['x_range']
        train_mu_range = params['train_mu_range']
        train_std_range = params['train_std_range']
        test_mu_range = params['test_mu_range']
        test_std_range = params['test_std_range']
        sideinfo = params['sideinfo']
        dataset_noise = params['dataset_noise']
        x_range = (np.array([x_range[0]]),np.array([x_range[1]]))
        phi_range = (np.array([train_mu_range[0],train_std_range[0]]),np.array([train_mu_range[1],train_std_range[1]]))
        func = ToyGaussianFunction(0,1.0,sideinfo=='derivative',sideinfo=='displacement')
        training_set = uniform_multi_task_sampler(func,x_range,phi_range,num_tasks=num_tasks,num_samples_per_task=num_samples_per_task)
        phi_range = (np.array([test_mu_range[0],test_std_range[0]]),np.array([test_mu_range[1],test_std_range[1]]))
        testing_set = uniform_multi_task_sampler(func,x_range,phi_range,num_tasks=num_tasks,num_samples_per_task=num_samples_per_task)
        training_set = add_dataset_noise(training_set,dataset_noise)
        testing_set = add_dataset_noise(testing_set,dataset_noise)
        val_split = int((1.0-val_split) * len(training_set))
        training_set, validation_set = training_set[:val_split], training_set[val_split:]
        return training_set, validation_set, testing_set


def get_dataset(std_condition = 'fixed', noise = 0.0, side_info_noise = 0.0):
    """Generates a panda dataframe of data"""
    result = pd.DataFrame(columns = ['idx', 'mu', 'std', 'y_max','xs','ys','derivatives','displacements'])

    mu_noise = noise
    derivative_noise = side_info_noise
    displacement_noise = side_info_noise
    num_samples_per_task=101
    mu_min=-7
    mu_max=7
    x_min = -10
    x_max = 10
    total_N_tasks = 90

    mus = np.linspace(mu_min,mu_max,total_N_tasks)
    xs = np.linspace(x_min,x_max,num_samples_per_task)
    h = 0.0001

    if std_condition =='fixed':
        stds = [1]*len(mus)
    elif std_condition == 'random':
        stds = 1.0 + np.numpy.random.rand(len(mus))
    counter = 0
    for (mu,std) in zip(mus,stds):
        pdf = scipy.stats.norm(mu, std)
        y_max = pdf.pdf(mu)
        ys = []
        derivatives = []
        displacements = []
        # ys = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu)/ std)**2) + np.random.normal(0, mu_noise, len(xs))
        # # Analytic derivative https://math.stackexchange.com/a/461154
        # derivatives = - (1 / ((std**3) * np.sqrt(2 * np.pi))) * xs * np.exp(-0.5 * ((xs - mu)/ std)**2) + np.random.normal(0, derivative_noise, len(xs))
        # displacements = xs - mu + np.random.normal(0, displacement_noise, len(xs))
        for x in xs:
            y = pdf.pdf(x) +  np.random.normal(0,mu_noise)
            ys.append(y)
            derivatives.append((pdf.pdf(x + h) - pdf.pdf(x))/h + \
                np.random.normal(0,derivative_noise))
            displacements.append(x - mu + np.random.normal(0,displacement_noise))
        #plt.scatter(np.array(xs),np.array(ys))
        #plt.show()
        #plt.scatter(np.array(xs),np.array(derivatives))
        #plt.show()
        #plt.scatter(np.array(xs),np.array(displacements))
        #plt.show()
        df = pd.DataFrame({'idx': counter, 'mu':mu, 'std':std, 'y_max':y_max, 'xs':list(xs),\
                            'ys':list(ys), 'derivatives':list(derivatives), 'displacements':list(displacements)})
        result = pd.concat([result, df], axis=0, join='outer')
        #result.append(df)
        counter += 1
    return result