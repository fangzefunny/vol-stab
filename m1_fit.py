import argparse 
import os 
import pickle
import datetime 
import numpy as np
import multiprocessing as mp
import pandas as pd

from sklearn.model_selection import KFold

from utils.model import subj
from utils.params import set_hyperparams
from utils.agents import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--fit_num', '-f', help='fit times', type = int, default=1)
parser.add_argument('--data_set', '-d', help='which_data', type = str, default='exp1_all')
parser.add_argument('--loss_fn', '-l', help='fitting methods', type = str, default='mle')
parser.add_argument('--group', '-g', help='fit to ind or fit to the whole group', type=str, default='ind')
parser.add_argument('--agent_name', '-n', help='choose agent', default='SMModel')
parser.add_argument('--cross_valid', '-k', help='do cross validatio or not', default=0)
parser.add_argument('--n_cores', '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=0)
parser.add_argument('--seed', '-s', help='random seed', type=int, default=2021)
args = parser.parse_args()
args.path = path

# create the folders if not existed
if not os.path.exists(f'{path}/fits'):
    os.mkdir(f'{path}/fits')
if not os.path.exists(f'{path}/fits/{args.agent_name}'):
    os.mkdir(f'{path}/fits/{args.agent_name}')

def fit_parallel( data, pool, model, verbose, args):
    '''A worker in the parallel computing pool 
    '''
    ## fix random seed 
    seed = args.seed
    n_params = len( args.bnds)

    ## Start fitting
    # fit cross validate 
    if args.cross_valid:
        kf = KFold( n_splits=5)
        params_lst = []
        nll = aic = bic = 0 
        for train_blks, test_blks in kf.split(data.keys()):
            m_data = np.sum([ data[ train_blk].shape[0] 
                           for train_blk in train_blks])
            train_data = { train_blk: data[ train_blk] 
                           for train_blk in train_blks}
            test_data  = { test_blk:  data[ test_blk] 
                           for test_blk  in  test_blks}
            opt_nll   = np.inf 
            results = [ pool.apply_async( model.fit, 
                            args=( train_data, args.bnds, args.bnds, 
                                    seed+2*i, verbose)
                            ) for i in range(args.fit_num)]
            for p in results:
                params, _ = p.get()
                loss = model.like_fn( params, test_data)
                if loss < opt_nll:
                    opt_nll, opt_params = loss, params 
                    # aic = 2*k + 2*nll 
                    opt_aic = n_params*2 + 2*nll
                    # bic = log(n)*k + 2*nll
                    opt_bic = n_params*m_data + 2*nll
            nll += opt_nll
            aic += opt_aic
            bic += opt_bic  
            params_lst.append( opt_params)

        # estimate params given cross-validated samples
        param_lst = np.vstack( params_lst)
        param_mean = np.mean( param_lst, axis=0)
        param_std  = np.std(  param_lst, aixs=0)
        fit_mat = np.vstack( np.hstack( [ param_mean, nll, aic, bic]),
                             np.hstack( [ param_std,  nll, aic, bic]))
    
    # if not cross-validated
    else: 
        m_data = np.sum([ data[key].shape[0] 
                           for key in data.keys()])
        results = [ pool.apply_async( model.fit, 
                        args=( data, args.bnds, args.bnds, 
                                seed+2*i, verbose)
                        ) for i in range(args.fit_num)]
        opt_nll   = np.inf 
        for p in results:
            params, loss = p.get()
            if loss < opt_nll:
                opt_nll, opt_params = loss, params  
                aic = n_params*2 + 2*opt_nll
                bic = n_params*m_data + 2*opt_nll
        fit_mat = np.hstack( [ opt_params, opt_nll, aic, bic]).reshape([ 1, -1])
        
    ## Save the params + nll + aic + bic 
    col = args.params_name + [ 'nll', 'aic', 'bic']
    fit_res = pd.DataFrame( fit_mat, columns=col)

    return fit_res 

def fit( data, args):
    '''Find the optimal free parameter for each model 
    '''
    ## Define the RL model 
    model = subj( args.agent)
    
    ## Start 
    start_time = datetime.datetime.now()
    
    ## Get the multiprocessing pool
    n_cores = args.n_cores if args.n_cores else int( mp.cpu_count()) 
    n_cores = np.min( [ n_cores, args.fit_num])
    pool = mp.Pool( n_cores)
    print( f'Using {n_cores} parallel CPU cores')

    ## Fit params to each individual 
    if args.group == 'ind':
        for sub_idx in data.keys():
            sub_data = { f'{sub_idx}': data[ sub_idx]}
            print( f'Fitting subject {sub_idx}')
            fit_res = fit_parallel( sub_data, pool, model, False, args)
            pname = f'{path}/fits/{args.agent_name}/params-{args.data_set}-{sub_idx}.csv'
            fit_res.to_csv( pname)

    ## Fit params to the population level
    elif args.group == 'avg':
        fit_res = fit_parallel( data, pool, model, True, args)
        pname = f'{path}/fits/{args.agent_name}/params-{args.data_set}-avg.csv'
        fit_res.to_csv( pname)
    
    ## END!!!
    end_time = datetime.datetime.now()
    print( '\nparallel computing spend {:.2f} seconds'.format(
            (end_time - start_time).total_seconds()))

def summary( data, args):

    ## Prepare storage
    n_sub    = len( data.keys())
    n_params = len( args.bnds)
    res_mat = np.zeros( [ n_sub, n_params+3]) + np.nan 
    res_smry = np.zeros( [ 2, n_params+3]) + np.nan 
    folder   = f'{path}/fits/{args.agent_name}'

    ## Loop to collect data 
    for i, sub_idx in enumerate( data.keys()):
        fname = f'{folder}/params-{args.data_set}-{sub_idx}.csv'
        log = pd.read_csv( fname, index_col=0)
        res_mat[ i, :] = log.iloc[ 0, :].values
        if i == 0: col = log.columns
    
    ## Compute and save the mean and sem
    res_smry[ 0, :] = np.mean( res_mat, axis=0)
    res_smry[ 1, :] = np.std( res_mat, axis=0) / np.sqrt( n_sub)
    fname = f'{path}/fits/params-{args.data_set}-{args.agent_name}-ind.csv'
    pd.DataFrame( res_smry, columns=col).to_csv( fname)

if __name__ == '__main__':

    ## STEP 0: LOAD DATA
    with open(f'{path}/data/{args.data_set}.pkl', 'rb') as handle:
        data = pickle.load( handle)

    ## STEP 1: HYPERPARAMETER TUNING
    args = set_hyperparams(args)   
            
    ## STEP 2: FIT
    fit( data, args)
    # summary the mean and std for parameters 
    if args.group == 'ind': summary( data, args)