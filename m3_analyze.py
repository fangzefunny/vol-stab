import os 
import pickle
import argparse 
import numpy as np
import pandas as pd 
import multiprocessing as mp

from utils.analyze import *
from utils.agents import *

# define path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--n_subj', '-f', help='f simulations', type=int, default=1)
parser.add_argument('--data_set', '-d', help='choose data set', default='exp1_rew')
parser.add_argument('--agent_name', '-n', help='choose agent', default='RDModel2')
parser.add_argument('--n_cores', '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=0)
args = parser.parse_args()

# create the folders for this folder
if not os.path.exists(f'{path}/analyses'):
    os.mkdir(f'{path}/analyses')

#=============================
#        Prep MP Pool
#=============================

def get_pool( args):
    n_cores = args.n_cores if args.n_cores else int( mp.cpu_count()) 
    print( f'Using {n_cores} parallel CPU cores')
    return mp.Pool( n_cores) 

#=============================
#        Quant Crieria
#=============================

def get_quant_criteria( data_set, model, sub_idx):
    fname = f'{path}/fits/{model}/params-{data_set}-{sub_idx}.csv'   
    record = pd.read_csv( fname, index_col=0)
    nll    = record.iloc[ 0, -3]
    aic    = record.iloc[ 0, -2]
    bic    = record.iloc[ 0, -1] 
    return nll, aic, bic 

def smry_quant_criteria( pool, outcomes, models, args):
    '''Summary three qualitative effects
        nll, aic, bic 
    '''
    ## Init the storage
    fname = f'{path}/data/{args.data_set}.pkl'
    with open( fname, 'rb') as handle:
        data = pickle.load( handle)
    subj_lst = list( data.keys())

    ## Get target criteria 
    for model in models:
        # check if the model has a record or not
        if model not in outcomes.keys(): outcomes[model] = dict()
        # start analyzing 
        print( f'Analyzing {model}')
        res = [ pool.apply_async( get_quant_criteria, 
            args=( args.data_set, model, subj_lst[i]))
            for i in range(len(subj_lst))]
        # unpack results
        nlls, aics, bics = [], [], []
        for p in res:
            crs = p.get()
            nlls.append( crs[0]) 
            aics.append( crs[1])
            bics.append( crs[2])
        outcomes[ model]['nll'] = np.mean( nlls)
        outcomes[ model]['aic'] = np.mean( aics)
        outcomes[ model]['bic'] = np.mean( bics)

    return outcomes 

#=============================
#     Analyze parameters
#=============================

def smry_params( outcomes, model_lst, args):
    '''Generate parameters for each model
    '''
    # get analyzable id 
    with open(f'{path}/data/{args.data_set}.pkl', 'rb') as handle:
        sub_ind = list(pickle.load( handle).keys())
    
    ## Loop to summary the feature for each model
    for model in model_lst:
        # check if the model has a record or not
        if model not in outcomes.keys(): outcomes[model] = dict()
        # start analyzing 
        print( f'Analyzing {model}')
        b_types, b_names = [ 1, 0], [ 'Stable', 'Volatile']
        temp = { 'alpha_s': [], 'alpha_a': [], 'beta': [],
                 'Trial type': []}
        for b, bn in zip( b_types, b_names):
            for sub_id in sub_ind:
                fname = f'{path}/fits/{model}/params-{args.data_set}-{sub_id}.csv'  
                data  = pd.read_csv( fname, index_col=0)
                if b:
                    temp['alpha_s'].append(data.iloc[ 0, 0])
                    temp['alpha_a'].append(data.iloc[ 0, 1]) 
                    temp['beta'].append(data.iloc[ 0, 2])
                else:
                    temp['alpha_s'].append(data.iloc[ 0, 3])
                    temp['alpha_a'].append(data.iloc[ 0, 4]) 
                    temp['beta'].append(data.iloc[ 0, 5])
                temp['Trial type'].append(bn)
        outcomes[model]['params'] = pd.DataFrame( temp)
    return outcomes

#==================================
#     Reward-Complexity analyses
#==================================

def get_RC_analyses( model, args):
    fname = f'{path}/simulations/{model}/sim_{args.data_set}-mode=reg.csv'
    data  = pd.read_csv( fname)
    subj_lst = data['sub_id'].unique()
    b_lst = [] 
    sub_lst = []
    if model != 'model11':
        outcomes = { 'rew': [], 'rew_hat': [], 
                     'EQ': [], 'pi_comp': [],}
    else:
        outcomes = { 'rew': [], 'rew_hat': []}
    b_types = [ 1, 0]
    b_names = [ 'Stable', 'Volatile']
    for subj in subj_lst:
        for b, bn in zip( b_types, b_names): 
            for key in outcomes.keys():
                idx = (data['sub_id'] == subj) &\
                      (data['b_type'] == b)
                outcomes[key].append( data[key][idx].mean())
            b_lst.append( bn)
            sub_lst.append( subj)
    outcomes['Trial type'] = b_lst
    outcomes['sub_id'] = sub_lst
    return pd.DataFrame( outcomes)

def smry_RC_analyses( outcomes, model_lst, args):
    
    ## Loop to summary the feature for each model
    for model in model_lst:
        # check if the model has a record or not
        if model not in outcomes.keys(): outcomes[model] = dict()
        # start analyzing 
        print( f'Analyzing {model}')
        res = get_RC_analyses( model, args)
        outcomes[model][f'RC-anlyses'] = res
    return outcomes

## Define a global Effect of interest  
if __name__ == '__main__':

    ## STEP0: GET THE COMPUTATIONAL POOL, 
    #  TARGET MODELS, AND CACHED ANALYSES 
    # computational pool
    n_cores = args.n_cores if args.n_cores else int( mp.cpu_count()*.8) 
    pool = mp.Pool( n_cores)
    print( f'Using {n_cores} parallel CPU cores')
    # target models
    models = args.agent_name.split(',')
    if '' in models: models.remove('')
    # cached analyses
    fname = f'{path}/analyses/analyses-{args.data_set}.pkl'
    try:
        with open( fname, 'rb')as handle:
            outcomes = pickle.load( handle)
    except:
        outcomes = dict() 

    ## STEP1: GET THE QUANTITATIVE METRICS
    outcomes = smry_quant_criteria( pool, outcomes, models, args)
    
    ## STEP2: GET RATE DISTORTION ANALYSES
    outcomes = smry_RC_analyses( outcomes, models, args)

    ## STEP3: GET PARAMS SUMMARY
    outcomes = smry_params( outcomes, models, args)
    
    ## STEP4: SAVE THE OUTCOMES
    with open( fname, 'wb')as handle:
        pickle.dump( outcomes, handle)