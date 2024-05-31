import warnings
import math
from typing import Union,List,Dict
import numpy as np
from .model import TrainableModel,AdaptiveModel, ProbabilisticModel,SupervisedLearningProblem
from .data import FewShotDataset, FlatDataset, MultiTaskDataset, TaskDataset
from . import utils
from tqdm import tqdm

def _assess(preds,classification=False,element=None):
    sum_se = 0.0
    sum_ae = 0.0
    sum_accuracy = 0
    sum_tp = 0
    sum_tn = 0
    sum_fp = 0
    sum_fn = 0
    n = 0
    for pred,y in preds:
        d = pred - y
        if element is not None:
            d = d[element]
        n_preds = 1
        if len(d.shape) == 2:
            #batch of predictions
            sum_se += np.sum([np.dot(d[:,i],d[:,i]) for i in range(d.shape[1])])
            n_preds = d.shape[0]
        else:
            sum_se += np.dot(d,d)
        sum_ae += np.sum(np.abs(d))
        if classification:
            #TODO: multi-class classification
            label = (pred > 0.5)
            gt_label = (y > 0.5)
            #TODO: batch of predictions
            if label == gt_label:
                sum_accuracy += 1
                if label:
                    sum_tp += 1
                else:
                    sum_tn += 1
            elif label:
                sum_fp += 1
            else:
                sum_fn += 1
        n += n_preds
    if classification:
        return {'RMSE':math.sqrt(sum_se/n),'MAE':sum_ae/n,'accuracy':sum_accuracy/n,'precision':sum_tp/(sum_tp+sum_fp),'recall':sum_tp/(sum_tp+sum_fn)}
    return {'RMSE':math.sqrt(sum_se/n),'MAE':sum_ae/n}




def fewshot_accuracy(model : Union[TrainableModel,AdaptiveModel,ProbabilisticModel],
                    test_data : MultiTaskDataset,
                    k_shot=0,classification=False,report_elements=False) -> Dict[str,float]:
    """Tests the accuracy of the given model on the given test data.
    
    If k_shot is given, then a k-shot model is assumed.  Accuracy of the model
    is tested after k support data are provided.

    Returns:
        dict: containing score names -> values 
    """
    preds = []
    if k_shot:
        if not isinstance(model,AdaptiveModel):
            warnings.warn("Testing non-adaptive model {} in non-adaptive mode".format(model.__class__.__name__))
        dataset = FewShotDataset(test_data,k_shot)
        for i in range(len(dataset)):
            supp_data,query_data = dataset[i]
            if hasattr(model,'update'):
                model.reset()
                for (x,y) in zip(*supp_data):
                    model.update(x,y)
            
            pred = model(query_data[0])
            if isinstance(model,ProbabilisticModel):
                assert len(pred)==2
                pred = pred[0]  #mean
            preds.append((pred,query_data[1]))

        if hasattr(model,'update'):
            model.reset()
    else:
        dataset = FlatDataset(test_data)
        for i in range(len(dataset)):
            x,y = dataset[i]
            pred = model(x)
            if isinstance(model,ProbabilisticModel):
                assert len(pred)==2,"Probabilistic model must return (mean,stddev) tuple, result is {}".format(pred)
                pred = pred[0]  #mean
            preds.append((pred,y))
    
    if report_elements or (len(pred) > 1 and len(pred) < 10):
        element_names = None
        if model.problem.output_vars and len(model.problem.output_vars)==len(pred):
            element_names = model.problem.output_vars
        else:
            element_names = ["element_"+str(i) for i in range(len(pred))]
        res = {}
        for i in range(len(pred)):
            res[element_names[i]] = _assess(preds,classification,i)
        res['overall'] = _assess(preds,classification)
        return res
    return _assess(preds,classification)


def fewshot_accuracy_incremental(model : Union[TrainableModel,AdaptiveModel,ProbabilisticModel],
                              test_data : MultiTaskDataset,
                              k_max=1,classification=False,report_elements=False) -> List[Dict[str,float]]:
    """Tests the accuracy of the given model on the given test data, running
    k-shot learning for k=0 to k_max.

    Returns:
        list of dict: for each k from 0 to k_max, the dict will contain a set
        of scores, mapping score names -> values.
    """
    preds = [[] for k in range(k_max+1)]
    if not isinstance(model,AdaptiveModel):
        warnings.warn("Testing non-adaptive model {} in non-adaptive mode".format(model.__class__.__name__))
    dataset = FewShotDataset(test_data,k_max,num_draws_per_task=10)
    print("Created few-shot dataset with",len(dataset),"task samples")
    
    for i in tqdm(range(len(dataset))):
        supp_data,query_data = dataset[i]
        if hasattr(model,'reset'):
            model.reset()
        for k in range(k_max+1):
            pred = model(query_data[0])
            if isinstance(model,ProbabilisticModel):
                assert len(pred)==2
                pred = pred[0]  #mean
            preds[k].append((pred,query_data[1]))
            if k < k_max and hasattr(model,'update'):
                model.update(supp_data[0][k],supp_data[1][k])
    if hasattr(model,'reset'):
        model.reset()
    n_out = pred.shape[-1]
    if report_elements or (n_out > 1 and n_out < 10):
        #give individual channels
        element_names = None
        if model.problem.output_vars and len(model.problem.output_vars)==n_out:
            element_names = model.problem.output_vars
        else:
            element_names = ["element_"+str(i) for i in range(n_out)]
        res = {}
        for i in range(n_out):
            res[element_names[i]] = [_assess(p,classification) for p in preds]
        res["overall"] = [_assess(p,classification) for p in preds]
        return res
    return [_assess(p,classification) for p in preds]
