import os 
import pickle
import argparse 
import numpy as np
import pandas as pd 
import multiprocessing as mp

from utils.analyze import *
from utils.agents import *
from m0_preprocess import split_data

# define path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--n_subj', '-f', help='f simulations', type=int, default=20)
parser.add_argument('--data_set', '-d', help='choose data set', default='collins_12')
parser.add_argument('--agent_name', '-n', help='choose agent', default='dual_sys')
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
    fname = f'{path}/data/{args.data_set}_subject.pkl'
    with open( fname, 'rb') as handle:
        data = pickle.load( handle)
    subj_lst = list( data.keys())

    ## Get target criteria 
    for model in models:
        # check if the model has a record or not
        if model not in outcomes.keys(): outcomes[model] == dict()
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
#     Other analyses
#=============================

## Define a global Effect of interest  
eoi = [ 'hc-pi_comp-subj', ' sz-pi_comp-subj',
        'hc-E_rew-subj',    'sz-E_rew_subj',
        'hc-pi_comp-block', 'sz-pi_comp-block',
        'hc-bias-block',    'sz-bias-block',
        'hc-bias-pi_comp',  'sz-bias-pi_comp',]

def get_analyses( data_set, model, sub_idx):
    fname = f'{path}/simulations/{model}/sim_{data_set}-idx{sub_idx}.csv'
    return Hutter_est( pd.read_csv( fname, index_col=0))

def smry_analyses( pool, outcomes, model_lst, args):
    '''Generate parameters for each model
    '''
    ## Init the stage
    outcomes = dict()

    ## Loop to summary the feature for each model
    for model in model_lst:
        # check if the model has a record or not
        if model not in outcomes.keys(): outcomes[model] == dict()
        # start analyzing 
        print( f'Analyzing {model}')
        res = [ pool.apply_async( get_quant_criteria, 
                args=( args.data_set, model, i)) 
                for i in range(args.n_subj)]
        # unpack results
        cum_crs = { eff: 0 for eff in eoi}
        for p in res:
            crs = p.get()
            for eff in eoi:
                cum_crs[eff] += crs[eff] / args.n_subj
        # record the result to the outcomes
        for eff in eoi:
            outcomes[eff] = cum_crs[eff] 
    
    return outcomes

if __name__ == '__main__':

    ## STEP0: GET THE COMPUTATIONAL POOL, 
    #  TARGET MODELS, AND CACHED ANALYSES 
    # computational pool
    n_cores = args.n_cores if args.n_cores else int( mp.cpu_count()*.8) 
    pool = mp.Pool( n_cores)
    print( f'Using {n_cores} parallel CPU cores')
    # target models
    models = args.agent_name.split(',')
    models.remove('')
    # cached analyses
    fname = f'{path}/summaries/analyses-{args.data_set}.pkl'
    try:
        with open( fname, 'rb')as handle:
            outcomes = pickle.load( handle)
    except:
        outcomes = dict() 
    
    ## STEP1: GET THE QUANTITATIVE METRICS
    outcomes = smry_quant_criteria( pool, outcomes, models, args)
    
    ## STEP2: GET OTHER ANALYSES
    outcomes = smry_analyses( pool, outcomes, models, args)
    
    ## STEP3: SAVE THE OUTCOMES
    with open( fname, 'wb')as handle:
        pickle.dump( outcomes, handle)