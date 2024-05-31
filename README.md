# Few-shot Regression and Sequential Decision Making Benchmark

Authors: Mengchao Zhang, Yifan Zhu, Dohun Jeong, Pranay Thangeda, Aditya Prakash, Patrick Naughton, and Kris Hauser

University of Illinois, Urbana-Champaign

## About

This package implements a PyTorch-based few-shot regression suite, as well as applications to
sequential decision-making problems with high-dimensional inputs.


## Installing dependencies

You will need the packages in requirements.txt (PyTorch, gPyTorch, Tensorboard, Scikit-learn)
```
> python -m pip install -r requirements.txt
```

## Few-shot regression

A few-shot learning model is defined as a function $y = f(x; x_{supp}, y_{supp})$ in which $x_{supp}$
is a list of $k$ inputs and $y_{supp}$ is a list of $k$ outputs already seen for the current task. 

We include the following core models:

Adaptive deep models:
- Deep Neural Network Fine Tuning (DNNFineTuning)
- Conditional Neural Processes (CNP) [3]
- BUGGY: Model-agnostic Meta Learning (MAML)
- Few-shot spatial regression [5]

Probabilistic, adaptive (shallow) models:
- Gaussian Process (GP), both independent and Multi-Output GP [6]
- Deep Neural Network Residual GP (DNNResidualGP)
- TODO: Deep Neural Network Residual MOGP (DNNResidualMOGP) 

Probabilistic, adaptive (deep) models:
- Deep Kernel Learning (DKL) [1]
- Deep Kernel Transfer (DKT) [2]
- Adaptive Deep Kernel Learning (ADKL) [4]


[1] Wilson, Andrew Gordon, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. "Deep kernel learning." In Artificial intelligence and statistics, pp. 370-378. PMLR, 2016.

[2] Patacchiola, Massimiliano, Jack Turner, Elliot J. Crowley, Michael O'Boyle, and Amos J. Storkey. "Bayesian meta-learning for the few-shot setting via deep kernels." Advances in Neural Information Processing Systems 33 (2020): 16108-16118.

[3] Garnelo, Marta, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and SM Ali Eslami. "Conditional neural processes." In International Conference on Machine Learning, pp. 1704-1713. PMLR, 2018.

[4] Tossou, Prudencio, Basile Dura, Francois Laviolette, Mario Marchand, and Alexandre Lacoste. "Adaptive deep kernel learning." arXiv preprint arXiv:1905.12131 (2019).

[5] Iwata, Tomoharu, and Yusuke Tanaka. "Few-shot learning for spatial regression via neural embedding-based Gaussian processes." Machine Learning (2021): 1-19.

[6] Bonilla, et al. "[Multi-task Gaussian Process Prediction](https://papers.nips.cc/paper_files/paper/2007/file/66368270ffd51418ec58bd793f2d9b1b-Paper.pdf)" Neural Information Processing Systems (2007).

In addition, we expect that each core model can customized with a "backbone" that encodes and decodes the output. These are described in [more detail below](#Input_and_output_encoders). 

### Basic usage

All the few-shot training for a model is handled for you inside of the `model.train` method.  You can then load and save using `model.load` and `model.save`.

```python
from fewshot import SupervisedLearningProblem,MultiTaskDataset,all_models

#TODO: create your data here with train,val,test splits (MultiTaskDataset objects)
train_dataset, val_dataset, test_dataset = create_my_multitask_dataset()
problem = SupervisedLearningProblem(input_shape=(n_in,),output_shape=(1,))
#can also import the model directly from fewshot.X
model_class = all_models()['DeepResidualGP']
m = model_class(problem)
#might override some default parameters here, or pass in a dict of parameters as the second argument to the model constructor
problem.params['k_shot']=5
problem.params['query_size']=100
problem.params['num_draws_per_task']=10

model.train(train_dataset,val_dataset)
model.save("my_model.pkl")
#evaluate the model -- either a single input or a batch will work
test_task = test_dataset[0]
test_x,test_y = test_task[0]  #data in task is stored as (x,y) pairs
print("Result:",model(test_x),"vs ground truth",test_y)
```

Some models (GP, DNNResidualGP) can also be trained in standard (non-few-shot) mode.  For these modes, you will not pass in the `k_shot` parameter.  Configuration files for these settings are found in the `configs/GP` and `configs/DNNResidualGP` folders, while the few-shot trained versions are in `configs/FSGP` and `configs/DNNResidualFSGP` folders.  Also, DKL is not few-shot trainable, so it should not be provided the `k_shot` parameter.  

For the online phase of few-shot learning, you will pass the support data to `model.update`.  You can reset the model using `model.reset`.

```python
test_task = test_dataset[0]
for k in range(5):
  supp_x,supp_y = test_task[k]
  model.update(supp_x,supp_y)
print("Result:",model(test_task[5][0]),"vs ground truth",test_task[5][1])
model.reset()
```

### Input and output encoders

Core models typically work on vector (tabular) inputs, but inputs are often images, text, etc. that 
need to be featurized into vector form. So, each model accepts an
**input encoder** that vectorizes the input x into a feature vector
of "reasonable" dimension.  Moreover, outputs may need to be featurized as well.  In this case
our models accept an **output encoder / decoder** that convert the output y into a feature vector.

The general architecture of the network is

`x -> input encoder -> core model -> output decoder -> y`

If any encoder / decoder is not provided, then it is treated as an identity function.

The loss function used for training is to compare the output of the core model to the encoded output.  In other words, we compare
`x -> input encoder -> core model -> core prediction` vs `encoded y <- output encoder <- y`

For non-deep GPs to work well (e.g., `GP`, `DNNResidualGP`), this dimension should ideally be on the order of
dozens.  However, GP models also have dimensionality reduction preprocessors that will reduce the dimension
if the encoding is too large.

Encoders and decoders are specified in the `SupervisedLearningProblem` object.  Relevant fields are:
- `x_encoder` (`torch.nn.Module`, default None)
- `y_encoder` (`torch.nn.Module`, default None)
- `y_decoder` (`torch.nn.Module`, default None)
- `encoded_input_shape` (`tuple`): the shape of the encoded input (currently must be length 1)
- `encoded_output_shape` (`tuple`): the shape of the encoded output (currently must be length 1)
- `x_encoder_trainable` (`bool`, default True): whether to train the input encoder
- `y_encoder_trainable` (`bool`, default False): whether to train the output encoder (not well tested.  No reconstruction loss.)
- `y_decoder_trainable` (`bool`, default False): whether to train the output decoder (not available for now)
- `train_on_encoded_x` (`bool`, default False): whether to preprocess the dataset by pre-encoding the input.  The input encoder will not be trained.
- `train_on_encoded_y` (`bool`, default False): whether to preprocess the dataset by pre-encoding the output.  The output encoder will not be trained.


### Hyperparameter configuration

Each method has a YAML file to configure its training hyperparameters.  Standard setups
are given in `configs/METHOD.yaml` where `METHOD` is the method name.  If you don't provide such
a file, the file in `fewshot/METHOD/config.yaml` is used.

Standard parameters include:

- seed: random seed for initializing network weights
- epochs: # of training epochs
- batch_size: batch size for dataloader
- num_workers: # of workers for dataloader
- optimizer: Adam
- lr: Initial learning rate
- lr_decay: Learning rate decay factor
- lr_step: # of epochs between learning rate decay
- patience: # of steps to wait for early stopping
- checkpoint_dir: Where intermediate 
- val_freq: # of steps between validation prints (if validation set provided)
- x_encoder: file containing pretrained weights of the x encoder
- y_decoder: file containing pretrained weights of the y encoder
- x_encoder_trainable: whether to train the x encoder or not
- y_encoder_trainable: whether to train the y encoder or not
- train_on_encoded_x: if x_encoder is not trainable, whether to pre-encode the training dataset (may save time)
- train_on_encoded_y: if y_encoder is not trainable, whether to pre-encode the training dataset (may save time)

For few-shot learning methods, the following parameters modify how the (support,query) pairs are drawn
- k_shot: an integer or list of integers specifying the few-shot training support sizes
- query_size: how many objecst to draw in the query size
- num_draws_per_task: how many samples are drawn per task
- task_batch_size: batch size of few-shot training

You may have to modify hyperparameters for a given problem. The best practice is to copy the
standard file into another directory *outside* of the `fewshot` folder and then modify it.  Config files
are easily specified in the `params` argument to each model constructor, or the `--modelconfig`
argument to `learning_test.py` and `decision_test.py`.

Note to implementers: Please don't make pushes to the config files inside the directory
unless you are really sure the default setup needs to change for ALL problems.


### Testing few-shot learning models

Run ``learning_test.py`` specifying a problem and a model as follows:

```
> python learning_test.py --problem toy_gaussian --model=DNNFineTuning --seed 0 --testsplit=0.3 
```

This will perform training and zero-shot testing of a model.  We specify the
random seed for consistency between runs.

To rerun the trained model but testing k-shot accuracy, use the ``--reload`` and
`--k_shot` arguments:

```
> python learning_test.py --problem toy_gaussian --model=DNNFineTuning --reload --seed 0 --testsplit=0.3 --k_shot=[0,1,2,3,4,5,10,20]
```

This will take much more time, since support sets from the entire dataset are sampled, and
then testing is performed on the remainder of the dataset.



Documentation:

```
usage: learning_test.py [-h] [--problem P] [--problemvariant C] [--model M]
               [--modelconfig C] [--reload] [--modelfile F]
               [--testsplit S] [--k_shot K] [--seed SEED]

A testing script for known models and problems in the fewshot library.

optional arguments:
  -h, --help         show this help message and exit
  --problem P        The benchmark problem to test, currently only supports
                     toy_gaussian
  --problemvariant C A variant provided in the problem configuration (see
                     items in the `variant` attribute in `fewshot/problems/PROBLEM_config.yaml`)
  --model M          The model to test, e.g., DNNResidualGP
  --modelconfig C    A YAML configuration file for the model, e.g.,
                     fewshot/DNNResidualGP/non-few-shot-config.yaml
  --reload           Reloads the previously trained temp model instead of
                     training
  --modelfile F      Reloads a previously trained model (.pkl) instead of
                     training
  --testsplit S      The fraction of the dataset to retain for testing, or
                     specify a 'standard' split by name
  --k_shot K         If metric=accuracy, a non-zero value performs k-shot
                     testing.  Can also provide a list
  --seed SEED        A random seed
```

Either ``model`` or ``modelconfig`` must be provided.

### Metrics

TODO: describe these


### Adding your own problem

To try these methods on your own supervised learning problem, you must:

1. Define the problem setting (`SupervisedLearningProblem`).
2. Define a dataset (`MultiTaskDataset`) consisting of data from multiple task instances.
3. Set up a desired method and train / test as usual.

### Existing models

Non-few-shot models (set model parameter `k_shot` to None)

- `configs/GP_flat.yaml`: a vanilla Gaussian Process.  Multiple output dimensions are predicted independently.
- `configs/MOGP_flat.yaml`: a Multi-Output Gaussian Process.  Correlations between output dimension are modeled.
- `configs/DNNFineTuning.yaml`: a Deep Neural Network, fine-tuned on the support set. 
- `configs/DNNResidualGP_flat.yaml`: a Residual Gaussian Process to model errors of the output of a DNN. 
 
Few-shot models (set model parameter `k_shot` to an integer or list of integers)

- `configs/GP_fs.yaml`: a Gaussian Process.   Multiple output dimensions are predicted independently.
- `configs/MOGP_fs.yaml`: a Multi-Output Gaussian Process.  Correlations between output dimension are modeled.
- `configs/DNNResidualGP_fs.yaml`: a Residual Gaussian Process to model errors of the output of a DNN.  
- `configs/DKL.yaml`: Deep Kernel Learning. Learns a deep feature map to a Gaussian Process.  
- `configs/DKT.yaml`: Deep Kernel Transfer. Learns a deep feature map to a Gaussian Process.  Trained in a task-by-task manner to model correlations between examples within a task.
- `configs/CNP.yaml`: Conditional Neural Process. Aggregates a latent vector predicted from the support set.  
- `configs/ADKL.yaml`: Adapative Deep Kernel Learning.  Aggregates a latent vector predicted from the support set and concatenates this to the deep feature map used in a DKL model.
- `configs/FewShotSR.yaml`: Few-shot Spatial Regression. 
- `configs/MAML.yaml`: Model-Agnostic Meta Learning. 


### Adding new models

A new model should subclass the `TorchTrainableModel` class, and optionally the
`AdaptiveModel` and `ProbabilisticModel` classes.

A model should accept configuration options via YAML files.  The convention is that they
should load from `fewshot/MODEL/config.yaml`.

When you are done implementing your model, add the model and its name to
the `all_models()` function in `fewshot/allmodels.py`.  It is now ready for testing
using the standard (`learning_test.py` / `decision_test.py`) scripts.


### Known issues

- When y is encoded/decoded, probabilistic models do not properly decode the standard deviation.
- iMAML is buggy.
- FewShotSR is buggy.


## Application to Sequential Decision Problems

Sequential decision making problems often involve transition dynamics, observation model, and even a state space that are unknown and/or difficult to model.  Such settings are commonly found in spatial / spatiotemporal active sensing problems, in which a latent and hidden field governs the observations and reward.  In these problems, the state space is the latent field itself, which is infinite dimensional and may have a complex relationship to the observations and reward.

A decision problem is modeled as a sequence `observation0, action0, outcome0, observation1, action1, outcome1, ...` of typically vector-like objects. There is a numeric reward that is a deterministic function `reward(observation,action,outcome)`, and the decision problem is to choose actions to maximize the sum of rewards. 

Few-shot regression is a useful approach to implicitly model the latent dynamics indirectly by simply learning the relationship from past actions and observations to future ones.  The approach uses incremental learning to *adapt on-line* to the history of observations and outcomes during each trial.  Specifically, after each `(observation,action,outcome,reward)` cycle, these items are encoded and passed to an `update()` method in the model.  The model then predicts a reward or an (outcome,reward) pair (depending on how our method is configured).

**Note: In the current stage of this work, we address stationary (contextual bandit-like) problems with latent characteristics. We are not currently addressing problems involving significant multi-step prediction and state dynamics.**

We usually model reward as the first element of the outcome vector, and indeed in many problems the only outcome is the reward.  The reason why we have an extra outcome vector is that a) some "extra" outcomes include action-conditioned sensor data (e.g., force readings or object displacements when manipulating an object). In many problems it would be cumbersome to include this information in the observation, e.g., would require a predictor to handle empty channels in the observation as special cases.

After the predictions are made, a score-based decision-maker estimates a scoring function `score(observation, action)` and at each stage picks the action that maximizes the score.  A Bayesian decision maker, such as UCB and Expected Improvement, provide a probability distribution of possible scores, and at each stage picks the action that maximizes some function of the score distribution.  For example, UCB picks $\arg \max_a E[s(o,a)] + s \cdot \sigma(o,a)$ where $\sigma(o,a)$ is the predicted standard deviation of the score and $s$ is a scaling factor that boosts the likelihood of choosing actions whose score is uncertain.

### Learning outcomes and rewards

We can configure the learning problem to either predict a reward `[reward] = f((observation,action))` or a concatenated outcome vector `(outcome,reward) = f((observation,action))`.

An optional input encoder consumes `(observation,action)` pairs and produces a vector $x(o,a)$.  An optional output encoder produces a vector $y(z)$ with $z$ denoting the `outcome` vector.  
The models will adapt to a support set $(X_{supp},Y_{supp})$ given by

$$X_{supp} = \\{x(o_{supp,1},a_{supp,1}),...,x(o_{supp,k},a_{supp,k}) \\},$$

$$Y_{supp} = \\{stack(y(z_{supp,1}),r_{supp,1}),...,stack(y(z_{supp,k}),r_{supp,k}) \\}.$$

The default encoders assume the observation, action, and outcome are vectors.  The default observation/action encoder just concatenates the vectors, i.e., $x_{default}(o,a)=stack(o,a)$, and the default outcome encoder $y_{default}(z)$ simply returns $z$.



### Testing decision-making models

Run ``decision_test.py`` specifying a problem and a model as follows:

```
> python decision_test.py --problem toy_gaussian --model=DNNFineTuning --seed 0 --testsplit=0.3 --metric_k_shot=5 --metric=accuracy
```

This will perform training and 0-shot accuracy testing of a model.  We specify the
random seed for consistency between runs.

To rerun the trained model but testing k-shot accuracy, use the ``--reload`` and
`--metric_k_shot` arguments:

```
> python test.py --problem toy_gaussian --model=DNNFineTuning --reload --seed 0 --testsplit=0.3 --metric_k_shot=5 --metric=accuracy
```

This will take much more time, since support sets from the entire dataset are sampled, and
then testing is performed on the remainder of the dataset.

Next, we can test in rollout mode using ``--metric=rollout``, which simulates what the
decision-maker would actually do in the benchmark:

```
> python test.py --problem toy_gaussian --model=DNNFineTuning --reload --seed 0 --testsplit=0.3 --metric_k_shot=5 --metric=rollout
```

By default, if the model is probabilistic, a UCB decision-maker is used. If the model
is not probabilistic, a greedy decision-maker is used.


Documentation:

```
usage: decision_test.py [-h] [--problem P] [--problemconfig C] --model M
               [--modelconfig C] [--reload] [--modelfile F]
               [--decisionmaker D] [--testsplit S] [--metric M]
               [--metric_k_shot K] [--seed SEED]

A testing script for known models and problems in the fewshot library.

optional arguments:
  -h, --help         show this help message and exit
  --problem P        The benchmark problem to test, currently only supports
                     toy_gaussian
  --problemconfig C  The config file used to specify a benchmark variant, e.g.
                     fewshot/problems/toy_gaussian_derivative_config.yaml
  --model M          The model to test, e.g., DNNResidualGP
  --modelconfig C    A YAML configuration file for the model, e.g.,
                     fewshot/DNNResidualGP/non-few-shot-config.yaml
  --reload           Reloads the previously trained temp model instead of
                     training
  --modelfile F      Reloads a previously trained model (.pkl) instead of
                     training
  --decisionmaker D  The decision making algorithm, e.g., Random, Greedy, UCB,
                     EI
  --testsplit S      The fraction of the dataset to retain for testing, or
                     specify a 'standard' split by name
  --metric M         Either 'accuracy' or 'rollout'
  --metric_k_shot K  If metric=accuracy, a non-zero value performs k-shot
                     testing
  --seed SEED        A random seed
```






### Adding your own problem

To try these methods on your own decision-making problem, you must:

1. Define the problem setting via a `DecisionTask`.
2. Define a dataset (`MultiTaskDataset`) consisting of data from multiple task instances.
3. Define a simulator for evaluation (`DecisionEnvironment`).

**Step 1**: You must implement a `DecisionTask` subclass that describes the setting
for your problem.  If your observation and/or outcome are not simple vectors, you should
specify the encoding backbone(s) as well.  (It is recommendd to have a default set of backbone
weights predefined, since most networks will not do a good job modifying these weights. )

**Step 2**: You must create a `MultiTaskDataset` for your problem defining data to train and
test models.

Each task is represented by a `TaskDataset` in which the dynamics of the system may depend
on some (latent) task variables.  These are like PyTorch Datasets in that they provide an abstract
list-like interface with `__len__` and `__getitem__`.  **NOTE: There is no assumed notion of sequentiality
in each task dataset.**

If you know the latent task variables, you can implement them in the `latent_vars` method.
(No implemented method uses these now, but in the future we might try to add methods that infer the
latent variables from the support set.)

**Step 3**: To perform evaluation, you should create a `DecisionEnvironment` subclass
that simulates your decision problem.  


### Adding new benchmarks

A new benchmark should subclass the `DecisionBenchmark` class to provide its task, training data, 
testing data, and (optional) testing environments.  By default, the testing environment
will just use the testing dataset.

Variants of benchmarks can be specified with YAML files.

When you are done implementing your benchmark, add the benchmark and its name to
the `all_benchmark()` function in `fewshot/decision/problems/allbenchmarks.py`.  It is now
ready for testing using the standard (test.py) script.


