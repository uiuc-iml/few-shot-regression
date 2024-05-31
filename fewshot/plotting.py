import numpy as np
from typing import List,Sequence,Tuple,Union
from .model import TrainableModel,AdaptiveModel, ProbabilisticModel,SupervisedLearningProblem
from .data import MultiTaskDataset
import matplotlib.pyplot as plt 

def plot_k_shot_preds(problem: SupervisedLearningProblem,
                        model : AdaptiveModel,
                        xs : np.ndarray,
                        support_set : List[Tuple[np.ndarray,np.ndarray]],
                        k_shot : Union[int,List[int],str]='auto',
                        x_dim : int = 0,
                        std_err_scale : float = 2):
    """Plots the predictions of a model given a support set.

    Result is a (fig,grid of matplotlib axes), which show the support data
    and predictions with the x axis being some varying dimension of the
    input.  The k-shot items are shown from left to right, and the y elements
    are shown from top to bottom.

    Args:
        problem: the problem
        model: the model to test
        xs: the x values to evaluate
        support_set: a list of x,y pairs to adapt to
        k_shot: if 'auto', tests all values of support_set.  Otherwise, if it's
            an int, tests 0-k items.  Otherwise, it must be a list of ints
            and each plot tests up to that value of support examples
        x_dim: the channel of the variable x axis in xs.
        std_err_scale: for probabilistic models, the scaling of the standard error
    """
    if k_shot == 'auto':
        k_shot = len(support_set)-1
    if isinstance(k_shot,int):
        k_shot = list(range(k_shot+1))
    assert isinstance(k_shot,list)
    assert len(xs) >= 2
    if len(k_shot) > 5:
        print("plot_k_shot_preds: Lots of shots, probably will get an ugly figure")
    xs = np.asarray(xs)
    if len(xs.shape) == 1:
        xs = xs.reshape((-1,1))
    assert x_dim < xs.shape[1], "Invalid x dimension specified"
    model.reset()
    ys = [model(x) for x in xs]
    nchannels = problem.output_shape[0]
    fig,axs = plt.subplots(nchannels,len(k_shot),figsize=(4*min(5,len(k_shot)),3*nchannels))
    if nchannels == 1:
        axs = [axs]
    k=0
    for i in range(len(k_shot)):
        new_supp_examples = []
        while k < k_shot[i]:
            xtrain,ytrain = support_set[k]
            new_supp_examples.append((xtrain,ytrain))
            model.update(xtrain,ytrain)
            #print("Updating",k," / ",k_shot[i])
            k+=1
        if len(new_supp_examples) > 0:
            ys = [model(x)  for x in xs]
            for dim in range(nchannels):
                axs[dim][i].scatter([x[x_dim] for x,y in new_supp_examples],[y[dim] for x,y in new_supp_examples],c='r',s=9)

        for dim in range(nchannels):
            ylabel = problem.output_vars[dim] if problem.output_vars is not None else 'y_'+str(dim+1)
            if i == 0:
                axs[dim][i].set_ylabel(ylabel)
            if isinstance(model,ProbabilisticModel):
                #show std errors
                ymeans = np.asarray([y[0][dim] for y in ys])
                ystds = np.asarray([y[1][dim] for y in ys])
                axs[dim][i].plot(xs[:,x_dim],ymeans,label=str(i)+'-shot')
                if std_err_scale>0:
                    axs[dim][i].fill_between(xs[:,x_dim],ymeans-std_err_scale*ystds,ymeans+std_err_scale*ystds,alpha=0.5)
            else:
                axs[dim][i].plot(xs[:,x_dim],ymeans,label=str(i)+'-shot')
        
        for dim in range(nchannels):
            if problem.input_vars is not None:
                xlabel = problem.input_vars[x_dim]
            else:
                xlabel = 'x_'+str(x_dim+1)
            axs[dim][i].set_xlabel(xlabel)
            #TODO: dynamically set y range
            axs[dim][i].set_ylim(-0.05,0.7)
            axs[dim][i].legend()
            
    return fig,axs

def plot_k_shot_accuracy(model : AdaptiveModel, test_set : MultiTaskDataset, k_shot=Union[int,List[int]], ax=None):
    """Plots the accuracy of a model on a test set.
    """
    from .metrics import fewshot_accuracy_incremental
    if ax is None:
        ax = plt.gca()
    if isinstance(k_shot, int):
        k_shot = list(range(k_shot+1))
    k_max = max(k_shot)
    accuracies = fewshot_accuracy_incremental(model,test_set,k_max=k_max,report_elements=True)
    #dictionary of (channel -> list of dict of (metric -> value))
    xs = np.asarray(k_shot)
    for k,metrics in accuracies.items():
        for metric in metrics[0].keys():
            ys = [metrics[i][metric] for i in k_shot]
            ax.plot(xs,ys,label=k + ' ' + metric)
    ax.legend()
    ax.set_xlabel('k shot')
