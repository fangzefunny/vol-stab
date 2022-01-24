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
parser.add_argument('--data_set', '-d', help='choose data set', default='rew_con')
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
    try:
        fname = f'{path}/fits/{args.agent_name}/params-rew_con-{sub_idx}.csv'  
    except:
        fname = f'{path}/fits/{args.agent_name}/params-rew_data_exp1-{sub_idx}.csv'  
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

eoi = [ 'alpha_s-block', 'alpha_a-block', 'beta-block', 
        'alpha_s-block', 'alpha_a-block', 'beta-block',]

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
        temp_dict = { eff: [[],[]] for eff in eoi}
        print( f'Analyzing {model}')
        for sub_id in sub_ind:
            try:
                fname = f'{path}/fits/{args.agent_name}/params-rew_con-{sub_id}.csv'  
            except:
                fname = f'{path}/fits/{args.agent_name}/params-rew_data_exp1-{sub_id}.csv'  
            data  = pd.read_csv( fname, index_col=0)
            temp_dict[f'alpha_s-block'][0].append(data.iloc[ 0, 0])
            temp_dict[f'alpha_s-block'][1].append(data.iloc[ 0, 3]) 
            temp_dict[f'alpha_a-block'][0].append(data.iloc[ 0, 1])
            temp_dict[f'alpha_a-block'][1].append(data.iloc[ 0, 4])
            temp_dict[f'beta-block'][0].append(data.iloc[ 0, 2])
            temp_dict[f'beta-block'][1].append(data.iloc[ 0, 5])
            #temp_dict[f'eq-{sub_type}-pi_comp'].append()
        # record the result to the outcomes
        for eff in eoi:
            outcomes[model][eff] = temp_dict[eff] 
    
    return outcomes

#==================================
#     Resource-rational analyses
#==================================

## Define a global Effect of interest  
eoi_rr = [ 'eq-pi_comp-Stab', 'eq-pi_comp-Vol',]

def get_rr_analyses( data_set, model, sub_idx):
    fname = f'{path}/simulations/{model}/sim_{data_set}-idx{sub_idx}.csv'
    data  = pd.read_csv( fname)
    subj_lst = data.sub_id.unique()
    crs = [ 'EQ', 'pi_comp']
    b_types = [ 1, 0]
    outcomes = []
    for b in b_types:
        phi = [[],[]]
        for i, cr in enumerate(crs):
            for subj in subj_lst:
                idx = (data['sub_id'] == subj) \
                    & (data['b_type'] == b) 
                cr_val = data[f'{cr}'][idx].mean()
                phi[i].append(cr_val)
        outcomes.append(phi)
    return outcomes

def smry_rr_analyses( pool, outcomes, model_lst, args):
    '''Generate parameters for each model
    '''
    ## Init the stage
    outcomes = dict()

    ## Loop to summary the feature for each model
    for model in model_lst:
        # check if the model has a record or not
        if model not in outcomes.keys(): outcomes[model] = dict()
        # start analyzing 
        print( f'Analyzing {model}')
        res = [ pool.apply_async( get_rr_analyses, 
                args=( args.data_set, model, i)) 
                for i in range(args.n_subj)]
        # unpack results
        cum_crs = { eff: 0 for eff in eoi_rr}
        for p in res:
            crs = p.get()
            cum_crs['eq-pi_comp-Stab'] += np.array(crs[0]) / args.n_subj 
            cum_crs['eq-pi_comp-Vol']  += np.array(crs[1]) / args.n_subj
        # record the result to the outcomes
        for eff in eoi_rr:
            outcomes[model][eff] = cum_crs[eff] 
    
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
    outcomes = smry_rr_analyses( pool, outcomes, models, args)

    ## STEP3: GET PARAMS SUMMARY
    outcomes = smry_params( outcomes, models, args)
    
    ## STEP4: SAVE THE OUTCOMES
    with open( fname, 'wb')as handle:
        pickle.dump( outcomes, handle)