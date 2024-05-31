import os
import ast
import json
import time
from fewshot import all_models
from fewshot.utils import set_seed, load_config
from fewshot.problems.allbenchmarks import all_benchmarks
from fewshot.metrics import fewshot_accuracy,fewshot_accuracy_incremental

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='A testing script for known models and problems in the fewshot library.')
    parser.add_argument('--problem', dest='problem', metavar='P', type = str, required=True, default='toy_gaussian', help="The benchmark problem to test, currently only supports toy_gaussian")
    parser.add_argument('--problemvariant', dest='problemvariant', metavar='C', type = str, help="The benchmark variant, e.g. out_distribution, with_derivative")
    parser.add_argument('--model', metavar='M', type = str, help="The model to test, e.g., DNNResidualGP")
    parser.add_argument('--modelconfig', dest='modelconfig', metavar='C', type = str, help="A YAML configuration file for the model, e.g., configs/DNNResidualGP_fs.yaml")
    parser.add_argument('--modelfile', dest='modelfile', metavar='F', type = str, default='', help="Reloads a previously trained model (.pkl) instead of training")
    parser.add_argument('--setting', dest='settings', metavar='S', action='append', help="Additional settings for the model, e.g., 'epochs=1'")
    parser.add_argument('--reload', dest='modelreload', action='store_true', help="Reloads the previously trained temp model instead of training")
    parser.add_argument('--k_shot', metavar="K", dest='k_shot', type=int, default=0, help="A non-zero value performs k-shot testing")
    parser.add_argument('--seed', dest='seed', type = int, help="A random seed")
    parser.add_argument('--device', dest='device', type = str, help="The PyTorch device to use, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument('--plot', dest='plot', type = str, default= 'False', help='If true, will plot and save figures during testing.')
    parser.add_argument('--output', dest='output', type = str, help='Output file (JSON) for results')

    args = parser.parse_args()
    problem = args.problem
    problemvariant = args.problemvariant
    modelname  = args.model
    modelreload = args.modelreload
    modelconfig = args.modelconfig
    modelfile = args.modelfile

    if not modelconfig and not modelname:
        print('Please provide a model name or a model configuration file')
        exit(1)

    seed = args.seed
    k_shot = args.k_shot
    plot = ast.literal_eval(args.plot)

    if seed is not None:
        set_seed(seed)
    
    t0 = time.time()
    #setting model config 
    if modelconfig:
        try:
            modelconfig = load_config(modelconfig)
        except:
            modelconfig = f'fewshot/{modelname}/config.yaml'
            print('Config file not found. Using default config from',modelconfig)
            modelconfig = load_config(modelconfig)
    else:
        modelconfig = f'fewshot/{modelname}/config.yaml'
        print('Using default config from',modelconfig)
        modelconfig = load_config(modelconfig)
    modelconfig['seed'] = seed
    if args.device:
        modelconfig['device'] = args.device

    if 'model' in modelconfig:  #get model name from config file 
        if modelname:
            if modelname != modelconfig['model']:
                print('Warning: model name in config file does not match the model name provided as argument')
                modelconfig['model'] = modelname
        else:
            modelname = modelconfig['model']
    else:
        modelconfig['model'] = modelname
    
    for setting in args.settings:
        key,val = setting.split('=')
        modelconfig[key] = ast.literal_eval(val)

    MODELS = all_models()
    if modelname not in MODELS:
        print("Model name '{}' is invalid, possible values:".format(modelname))
        for name in MODELS.keys():
            print(" ",name)
        exit(1)
    selected_model = MODELS[modelname]
    
    BENCHMARKS = all_benchmarks()
    if problem not in BENCHMARKS:
        print("Problem '{}' is invalid, possible values:".format(problem))
        for name in BENCHMARKS.keys():
            print(" ",name)
        exit(1)
    
    benchmark = BENCHMARKS[problem](variant=problemvariant)
    learning_problem = benchmark.problem()
    print("Time for creating problem and model:",time.time()-t0)

    print("Creating dataset...")
    t0 = time.time()
    train_data,val_data,test_data = benchmark.dataset()
    print("Time to create dataset of",sum([len(train_data[i]) for i in range(len(train_data))]),"training samples:",time.time()-t0)

    #extract benchmark parameters
    model = selected_model(learning_problem,modelconfig)
    default_model_fn = 'trained_models/test_{}_{}.pkl'.format(problem,modelname)
    if modelfile:
        if os.path.exists(modelfile):
            # go straight to test
            model.load(modelfile)
        else:
            raise ValueError('Model file not found: {}'.format(modelfile))
    elif modelreload:
        model.load(default_model_fn)
    else:
        print('Start model training... ')
        model.train(train_data, val_data)
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        print("Saving model to",default_model_fn)
        model.save(default_model_fn)

    print("Beginning evaluation...")
    attempts_counts = []
    success_rates = []
    if k_shot:
        res = fewshot_accuracy_incremental(model,test_data,k_shot)
    else:
        res = fewshot_accuracy(model,test_data,k_shot)
    if args.output:
        print("Saving results in JSON format to",args.output)
        with open(args.output,'w') as f:
            json.dump(res,f,indent=2)
    else:
        print(json.dumps(res, indent=2))