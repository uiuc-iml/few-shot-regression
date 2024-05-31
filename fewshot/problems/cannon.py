import numpy as np
import pandas as pd
import math
from ..data import MultiTaskDataset, StandardTaskDataset, uniform_multi_task_sampler, add_dataset_noise
from .. import utils
from ..model import SupervisedLearningProblem
from .benchmark import LearningBenchmark

g = 9.81

def cannon_shooting_result(init_speed, theta, wind_speed):
    vx = math.cos(math.radians(theta))*init_speed + wind_speed
    vy = math.sin(math.radians(theta))*init_speed
    return 2*vx*vy/g


class CannonShootingFunc:
    def __init__(self, wind_speed, target, include_sign = True):
        self.wind_speed = wind_speed
        self.target = target
        self.include_sign = include_sign
    def __call__(self, action, task_parameters = None):
        init_speed, theta = action
        if task_parameters is not None:
            wind_speed, target = task_parameters
        else:
            wind_speed, target = self.wind_speed, self.target
        distance = cannon_shooting_result(init_speed, theta, wind_speed)
        reward = -np.abs(distance - target)
        if self.include_sign:
            side_info = np.sign(distance - target)
            return np.array([reward, side_info])
        return np.array([reward])

class CannonBenchmark(LearningBenchmark):
    def __init__(self, variant=None):
        import os
        config_file = os.path.join(os.path.dirname(__file__), 'cannon_config.yaml')
        self.params = utils.load_config(config_file, variant)

    def problem(self):
        func = CannonShootingFunc(0,0)
        n_out = len(func([0.0,0.0]))
        output_vars = ['reward']
        if n_out > 1:
            output_vars.append('side_info')
        return SupervisedLearningProblem(input_shape=(2,),output_shape=(n_out,),input_vars=['init_speed','theta'],output_vars=output_vars)

    def dataset(self,val_split=0.2):
        params = self.params
        num_tasks = params['num_tasks']
        num_samples_per_task = params['num_samples_per_task']
        init_speed_range = params['init_speed_range']
        theta_range = params['theta_range']
        train_wind_speed_range = params['train_wind_speed_range']
        train_target_range = params['train_target_range']
        test_wind_speed_range = params['test_wind_speed_range']
        test_target_range = params['test_target_range']
        dataset_noise = params['dataset_noise']

        x_range = (np.array([init_speed_range[0],theta_range[0]]),np.array([init_speed_range[1],theta_range[1]]))
        phi_range = (np.array([train_wind_speed_range[0],train_target_range[0]]),np.array([train_wind_speed_range[1],train_target_range[1]]))
        func = CannonShootingFunc(0,0)
        training_set = uniform_multi_task_sampler(func,x_range,phi_range,num_tasks=num_tasks,num_samples_per_task=num_samples_per_task)
        phi_range = (np.array([test_wind_speed_range[0],test_target_range[0]]),np.array([test_wind_speed_range[1],test_target_range[1]]))
        testing_set = uniform_multi_task_sampler(func,x_range,phi_range,num_tasks=num_tasks,num_samples_per_task=num_samples_per_task)
        training_set = add_dataset_noise(training_set,dataset_noise)
        testing_set = add_dataset_noise(testing_set,dataset_noise)
        val_split = int((1.0-val_split) * len(training_set))
        training_set, validation_set = training_set[:val_split], training_set[val_split:]
        return training_set, validation_set, testing_set


def get_dataset(noise=0.0,g=9.81):
    """Generates a panda dataframe of data where a datapoint represents the projectile distance 
    for a given action under a given wind speed"""

    result = pd.DataFrame(columns = ['idx', 'wind_speed','actions','distances'])

    wind_speed_min=0
    wind_speed_max=3
    init_speed_min = 4
    init_speed_max = 6
    theta_min = 0
    theta_max = 90

    wind_speeds = np.linspace(wind_speed_min,wind_speed_max,31)
    init_speeds = np.linspace(init_speed_min,init_speed_max,21)
    thetas = np.linspace(theta_min,theta_max,91)
        
    counter = 0
    for wind_speed in wind_speeds:
        distances = []
        actions = []
        for init_speed in init_speeds:
            for theta in thetas:
                dist = cannon_shooting_result(init_speed, theta, wind_speed)
                distances.append(dist + np.random.normal(0,noise))
                actions.append([init_speed,theta])
            
        df = pd.DataFrame({'idx': counter, 'wind_speed':wind_speed, 'actions':actions, 'distances':distances})
        result = pd.concat([result, df], axis=0, join='outer')
        counter += 1
    return result