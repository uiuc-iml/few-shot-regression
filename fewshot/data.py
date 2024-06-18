from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List,Tuple,Any,Optional,Union,Callable
import numpy as np
import torch

import time

class TaskDataset:
    """A supervised learning dataset on a single task."""
    @abstractmethod
    def __len__(self) -> int:
        """Returns the # of instances for this task."""
        raise NotImplementedError()
    
    @abstractmethod
    def __getitem__(self,i:int) -> Tuple[Any,Any]:
        """Returns an (x,y) tuple. """
        raise NotImplementedError()
    
    def latent_vars(self):
        """If not None, gives ground-truth latent variables for this task."""
        return None

    def split(self,fraction=0.5,shuffle=True) -> Tuple[TaskDataset,TaskDataset]:
        """Splits a dataset into two pieces (task-wise). If shuffle=True, the
        indices are chosen randomly.
        """
        if shuffle:
            order = np.random.permutation(len(self))
        else:
            order = list(range(len(self)))
        splitindex = int(len(order)*fraction)
        return SubsetTaskDataset(self,order[:splitindex]),SubsetTaskDataset(self,order[splitindex:])

    def subsample(self,fraction=None,count=None) -> StandardTaskDataset:
        """Subsamples the dataset.  If fraction is given, this is the fraction
        of examples to keep.  If count is given, this is the total number of
        examples to keep.
        """
        if count is not None:
            count = min(count,len(self))
        elif fraction is not None:
            count = int(fraction*len(self))
        if count==len(self):
            return self
        indices = np.sort(np.random.choice(len(self),count,replace=False))
        return SubsetTaskDataset(self,indices)


class StandardTaskDataset(TaskDataset):
    """A simple list of (x,y) pairs."""
    def __init__(self,items, latent_vars=None):
        self.items = items
        self._latent_vars = latent_vars
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self,i:int) -> Tuple[Any,Any]:
        if isinstance(i,slice):
            return StandardTaskDataset([self.items[j] for j in range(*i.indices(len(self)))])
        return self.items[i]
    
    def latent_vars(self):
        return self._latent_vars


class SubsetTaskDataset(TaskDataset):
    """Subsets a set of tasks."""
    def __init__(self,data : TaskDataset,indices : List[int]):
        self.data = data
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, i:int):
        return self.data[self.indices[i]]

    def latent_vars(self):
        return self.data.latent_vars()


def AugmentedTaskDatset(TaskDataset):
    """A task dataset that performs data augmentation on the fly.  The
    transform function should take an (x,y) pair and return a new (x,y) pair."""
    def __init__(self, data : TaskDataset, transform : Callable):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        x,y = self.data[i]
        return self.transform(x,y)

    def latent_vars(self):
        return self.data.latent_vars()


class MultiTaskDataset(ABC):
    """A multi-task decision dataset."""
    @abstractmethod
    def __len__(self) -> int:
        """Returns the # of tasks."""
        raise NotImplementedError()
    
    @abstractmethod
    def __getitem__(self,i) -> TaskDataset:
        """Returns the TaskDataset representing the i'th task. """
        raise NotImplementedError()

    def split(self,fraction=0.5,shuffle=True) -> Tuple[MultiTaskDataset,MultiTaskDataset]:
        """Splits a dataset into two pieces (task-wise). If shuffle=True, the
        indices are chosen randomly.
        """
        if shuffle:
            order = np.random.permutation(len(self))
        else:
            order = list(range(len(self)))
        splitindex = int(len(order)*fraction)
        return SubsetMultiTaskDataset(self,order[:splitindex]),SubsetMultiTaskDataset(self,order[splitindex:])

    def subsample(self,fraction=None,example_fraction=None,task_count=None,example_count=None):
        """Subsamples the dataset.  If fraction is given, this is the fraction
        of tasks to keep.  If example_fraction is given, this is the fraction of
        examples to keep.  If task_count is given, this is the number of tasks
        to keep.  If example_count is given, this is the total number of examples 
        to keep.
        """
        if task_count is not None:
            task_count = min(task_count,len(self))
        elif fraction is not None:
            task_count = int(fraction*len(self))
        else:
            task_count = len(self)
        task_indices = np.sort(np.random.choice(len(self),task_count,replace=False))
        tasks = [self[i] for i in task_indices]
        if example_count is not None:
            example_fraction = example_count/sum(len(task) for task in tasks)
        if example_fraction is None or example_fraction >= 1.0:
            return StandardMultiTaskDataset(tasks)
        return StandardMultiTaskDataset([task.subsample(fraction=example_fraction) for task in tasks])
            

class StandardMultiTaskDataset(MultiTaskDataset):
    """Just a list of TaskDatasets"""
    def __init__(self,items):
        self.items = items
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, i : int) -> TaskDataset:
        if isinstance(i,slice):
            return StandardMultiTaskDataset([self.items[j] for j in range(*i.indices(len(self)))])
        return self.items[i]


class SubsetMultiTaskDataset(MultiTaskDataset):
    """Subsets a set of tasks."""
    def __init__(self,data : MultiTaskDataset,task_indices : List[int]):
        self.data = data
        self.task_indices = task_indices
    
    def __len__(self) -> int:
        return len(self.task_indices)
    
    def __getitem__(self, i:int) -> TaskDataset:
        return self.data[self.task_indices[i]]


class AugmentedMultiTaskDataset(MultiTaskDataset):
    """A multi-task dataset that performs data augmentation on the fly.  The
    transform function should take an (x,y) pair and return a new (x,y) pair.
    """
    def __init__(self, data : MultiTaskDataset, transform : Callable):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return AugmentedTaskDatset(self.data[i],self.transform)


class FlatDataset:
    """Converts a multi-task dataset to a Pytorch-compatible dataloader
    for standard learning.
    
    Works with data augmentation and dynamic loading.
    """
    def __init__(self,data : MultiTaskDataset):
        self.data = data
        self.all_data_indices = []
        for i,task in enumerate(data):
            for j in range(len(task)):
                self.all_data_indices.append((i,j))
    
    def __len__(self):
        return len(self.all_data_indices)
        #return sum(len(task) for task in self.data)
    
    def __getitem__(self,i:int) -> Tuple[Any,Any]:
        t,j = self.all_data_indices[i]
        assert t < len(self.data)
        assert j < len(self.data[t])
        return self.data[t][j]


class TaskBatchedDataset:
    """Converts a multi-task dataset into a batched Pytorch-compatible
    dataset sampled for standard learning.  Can iterate through this
    directly, or you can use a Dataloader with batchsize=1.

    For each task, will draw `num_draws_per_task` tuples `(xb,yb)` each
    of leading dimension `batch_size`.  Each batch is guaranteed to consist
    of examples from the same task.  If `num_draws_per_task` is None (default),
    then the task is split evenly into batches.
    """
    def __init__(self, data : MultiTaskDataset, batch_size : int,
                      num_draws_per_task : Optional[Union[int,float]]=None,
                      shuffle=True):
        self.data = data
        self.batch_size = batch_size
        item_indices = []
        for i in range(len(self.data)):
            task = self.data[i]
            ndraws = num_draws_per_task
            if num_draws_per_task is None:
                if shuffle:
                    tindices = np.random.permutation(len(task))
                else:
                    tindices = np.arange(len(task))
                #split task evenly
                for b in range(0,len(task),batch_size):
                    if b + batch_size <= len(task):
                        item_indices.append((task,tindices[b:b+batch_size]))
                    else:
                        items = tindices[b:].tolist()
                        for j in range(batch_size - (len(task)-b)):
                            items.append(np.random.randint(0,len(task)-1))
                        item_indices.append((task,np.array(items)))
            else:
                if isinstance(num_draws_per_task,int):
                    ndraws = num_draws_per_task
                elif isinstance(num_draws_per_task,float):
                    ndraws = int(len(task)*num_draws_per_task)
                for draw in range(ndraws):
                    #sample a batch from the task
                    k = min(self.batch_size,len(task)-1)
                    inds = np.arange(len(task))
                    np.random.shuffle(inds)
                    batch = inds[:k]
                    item_indices.append((task,batch))
        self.item_indices = item_indices    # List[Tuple[TaskDataset,List[int]]]
    
    def __len__(self):
        return len(self.item_indices)
    
    def __getitem__(self,i) -> Tuple[Any,Any]:
        task,batch = self.item_indices[i]
        xb = []
        yb = []
        for i in batch:
            if len(task[i])==2:
                xb.append(task[i][0])
                yb.append(task[i][1])
            else:
                xb.append(task[i][:-1])
                yb.append(task[i][-1])
        return (torch.utils.data.default_collate(xb),torch.utils.data.default_collate(yb))


class FewShotDataset:
    """Converts a multi-task dataset into a Pytorch-compatible dataset
    sampled for few-shot learning.  
    
    For each task, will draw `num_draws_per_task` tuples
    `((x_supp,y_supp),
      (x_query,y_query))`
    by sampling `support_size` items from each task for the support set, and
    `query_size` items for the query set.

    Note:
        During training, must set `batch_size=1` if `support_size` is a list.

    Args:
        data: the base dataset
        support_size (int or list of int): the size of each support set, list
            of sizes to draw from.
        query_size (int, optional): if given, the # of query items drawn per
            sample.  Otherwise, all the non-support data in a task will be used
            as the query set.
        num_draws_per_task (int or float, optional): if None, `len(task)` draws
            will be performed per task.  If int, this many draws will be
            performed per task. If float, `len(task)*num_draws_per_task` draws
            will be done per task.
            
    """
    def __init__(self,data : MultiTaskDataset, support_size : Union[int,List[int]],
                      query_size : Optional[int]=None,
                      num_draws_per_task : Optional[Union[int,float]]=None):
        self.data = data
        self.support_size = support_size
        self.query_size = query_size
        item_indices = []
        for i in range(len(self.data)):
            task = self.data[i]
            ndraws = num_draws_per_task
            if num_draws_per_task is None:
                ndraws = len(task)
            elif isinstance(num_draws_per_task,int):
                ndraws = num_draws_per_task
            elif isinstance(num_draws_per_task,float):
                ndraws = int(len(task)*num_draws_per_task)
            for draw in range(ndraws):
                #sample a support set from the task
                if isinstance(self.support_size,int):
                    k = min(self.support_size,len(task)-1)
                else:
                    k = np.random.choice(self.support_size)
                    k = min(k,len(task)-1)
                inds = np.arange(len(task))
                np.random.shuffle(inds)
                supp = inds[:k]
                if self.query_size is None:
                    query = inds[k:]
                else:
                    query = inds[k:min(k+self.query_size,len(inds))]
                item_indices.append((task,supp,query))
        self.item_indices = item_indices    # List[Tuple[TaskDataset,List[int],List[int]]]
    
    def __len__(self):
        return len(self.item_indices)
    
    def __getitem__(self,i) -> Tuple[Tuple[Any,Any],Tuple[Any,Any]]:
        task,supp,query = self.item_indices[i]
        supp_x = []
        supp_y = []
        query_x = []
        query_y = []
        for i in supp:
            if len(task[i])==2:
                supp_x.append(task[i][0])
                supp_y.append(task[i][1])
            else:
                supp_x.append(task[i][:-1])
                supp_y.append(task[i][-1])
        for i in query:
            if len(task[i])==2:
                query_x.append(task[i][0])
                query_y.append(task[i][1])
            else:
                query_x.append(task[i][:-1])
                query_y.append(task[i][-1])
        return ((stack(supp_x),stack(supp_y)),(stack(query_x),stack(query_y)))



def stack(data):
    if len(data)==0:
        return data
    e = data[0]
    if isinstance(e,(int,float)):
        return torch.tensor(data)
    if isinstance(e,torch.Tensor):
        return torch.stack(data)
    if isinstance(e,np.ndarray):
        return np.stack(data)
    print("Data is not stackable")
    return data

def transformed_dataset(dataset : Union[list,TaskDataset,MultiTaskDataset,FewShotDataset],
                        input_transform : Optional[Callable]=None,
                        output_transform : Optional[Callable]=None):
    """Returns a new dataset that applies a transformation to the inputs and
    outputs.  If a transform is None, the original data is returned.
    """
    if isinstance(dataset,FewShotDataset):
        res = FewShotDataset(transformed_dataset(dataset.data),dataset.support_size,dataset.query_size,0)
        res.item_indices = dataset.item_indices
        return res
    elif isinstance(dataset,MultiTaskDataset):
        return StandardMultiTaskDataset([transformed_dataset(d,input_transform,output_transform) for d in dataset])
    else:
        transformed = []
        for i in range(len(dataset)):
            x,y = dataset[i]
            if input_transform is not None:
                x = input_transform(x)
            if output_transform is not None:
                y = output_transform(y)
            transformed.append((x,y))
        if isinstance(dataset,TaskDataset):
            return StandardTaskDataset(transformed,dataset.latent_vars())
        return transformed

def uniform_task_sampler(func : Callable, x_range : Tuple[np.ndarray,np.ndarray], num_samples : int) -> TaskDataset:
    """A simple task dataset sampler.  Given a function
    `f(x) -> y` and a range of x values, this will sample a TaskDataset 
    uniformly at random
    """
    items = []
    x = np.random.uniform(x_range[0],x_range[1],(num_samples,len(x_range[0])))
    for j in range(num_samples):
        y = func(x[j])
        items.append((x[j],y))
    return StandardTaskDataset(items)

def uniform_multi_task_sampler(func : Callable, x_range : Tuple[np.ndarray,np.ndarray], task_parameter_range : Tuple[np.ndarray,np.ndarray],
                              num_tasks : int, num_samples_per_task : int) -> MultiTaskDataset:
    """A simple multi-task dataset sampler.  Given a function
    `f(x,task_parameters) -> y`, a range of x values, and a range of task
    parameters, this will sample a MultiTaskDataset uniformly at random.
    """
    tasks = []
    ttotal = 0.0
    ntotal = 0
    task_parameters = np.random.uniform(task_parameter_range[0],task_parameter_range[1],(num_tasks,len(task_parameter_range[0])))
    for i in range(num_tasks):
        items = []
        x = np.random.uniform(x_range[0],x_range[1],(num_samples_per_task,len(x_range[0])))
        t0 = time.time()
        for j in range(num_samples_per_task):
            y = func(x[j],task_parameters[i])
            items.append((x[j],y))
        t1 = time.time()
        ttotal += (t1-t0)
        ntotal += num_samples_per_task
        task = StandardTaskDataset(items)
        tasks.append(task)
    if ttotal > 5.0:
        print("uniform_multi_task_sampler: Sampling time is",ttotal/ntotal,"s per item,",ntotal,"items")
    return StandardMultiTaskDataset(tasks)

def add_dataset_noise(dataset : Union[TaskDataset,MultiTaskDataset], noise : float) -> Union[TaskDataset,MultiTaskDataset]:
    """Adds noise to a multi-task dataset."""
    if isinstance(dataset, MultiTaskDataset):
        tasks = []
        for task in dataset:
            tasks.append(add_dataset_noise(task,noise))
        return StandardMultiTaskDataset(tasks)
    else:
        assert isinstance(dataset, TaskDataset)
        items = []
        for i in range(len(dataset)):
            x,y = dataset[i]
            y = y + np.random.normal(0,noise,y.shape)
            items.append((x,y))
        return StandardTaskDataset(items)

def pandas_to_task_dataset(df, x_columns : List[Union[int,str]], y_columns : List[Union[int,str]]) -> TaskDataset:
    """Converts a pandas dataframe to a TaskDataset."""
    items = []
    for i in range(len(df)):
        x = df.iloc[i][x_columns].values
        y = df.iloc[i][y_columns].values
        items.append((x,y))
    return StandardTaskDataset(items)

def pandas_to_multitask_dataset(df, task_columns : List[Union[int,str]],
                                    x_columns : List[Union[int,str]],
                                    y_columns : List[Union[int,str]]) -> MultiTaskDataset:
    """Converts a pandas dataframe to a MultiTaskDataset.  All the columns
    whose task_columns are equal are grouped into a single task.
    """
    tasks = {}
    for i in range(len(df)):
        task = tuple(df.iloc[i][task_columns].values)
        if task not in tasks:
            tasks[task] = []
        x = df.iloc[i][x_columns].values
        y = df.iloc[i][y_columns].values
        tasks[task].append((x,y))
    return StandardMultiTaskDataset([StandardTaskDataset(items,latent) for latent,items in tasks.items()])
