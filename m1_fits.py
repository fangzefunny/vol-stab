import argparse 
import os 
import pickle
import datetime 
import pandas as pd
# multi processing 
import multiprocessing as mp

from utils.model import subj
from utils.hyperparams import set_hyperparams
from utils.brains import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--fit_num', '-f', help='fit times', type = int, default=1)
parser.add_argument('--brain_name', '-n', help='choose agent', default='RRmodel')
parser.add_argument('--n_cores', '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=0)
parser.add_argument('--seed', '-s', help='random seed', type=int, default=2021)
args = parser.parse_args()
args.path = path

def fit( train_data, args):

    # define the RL2 model 
    model = subj( args.brain)

    start_time = datetime.datetime.now()
       
    if args.n_cores:
        n_cores = args.n_cores
    else:
        n_cores = int( mp.cpu_count())
    n_cores = np.min( [ n_cores, args.fit_num])
    pool = mp.Pool( n_cores)
    
    for sub_idx in train_data.keys():

        # get subject data
        sub_data = [train_data[ sub_idx]]
        ## Start fitting 
        print( f'Using {n_cores} parallel CPU cores')
        # parameter matrix and loss matrix 
        fit_mat = np.zeros( [args.fit_num, len(args.bnds) + 1])
        # param
        seed = args.seed
        results = [ pool.apply_async( model.fit, args=(sub_data, args.bnds, seed+2*i)
                        ) for i in range(args.fit_num)]
        for i, p in enumerate(results):
            param, loss  = p.get()
            fit_mat[ i, :-1] = param
            fit_mat[ i,  -1]  = loss
        end_time = datetime.datetime.now()
        print( f'\nparallel computing spend {(end_time - start_time).total_seconds():.2f} seconds')
        # choose the best params and loss 
        loss_vec = fit_mat[ :, -1]
        opt_idx, loss_opt = np.argmin( loss_vec), np.min( loss_vec)
        param_opt = fit_mat[ opt_idx, :-1]

        # save the fit results
        col = args.params_name + ['loss']
        fit_results = pd.DataFrame( fit_mat, columns=col)

        # save fit results
        fname = f'{path}/results/fit_results-{args.brain_name}-{sub_idx}.csv'
        
        # save fitted parameter 
        try:
            fit_results.to_csv( fname)
        except:
            os.mkdir( f'{path}/results')
            fit_results.to_csv( fname)

        # save opt params 
        params_mat = np.zeros( [1, len(args.bnds) + 1])
        params_mat[ 0, :-1] = param_opt
        params_mat[ 0, -1]  = loss_opt
        col = args.params_name + ['mle_loss']
        params = pd.DataFrame( params_mat, columns=col)
        
        # create filename 
        fname = f'{path}/results/params-{args.brain_name}-{sub_idx}.csv'
        params.to_csv( fname)

        break 

if __name__ == '__main__':

    ## STEP 0: LOAD DATA
    with open(f'{path}/data/processed_exp1.pkl', 'rb') as handle:
        train_data = pickle.load( handle)

    ## STEP 1: HYPERPARAMETER TUNING
    args = set_hyperparams(args)   
            
    ## STEP 2: FIT TO EACH SUBJECT
    fit( train_data, args)
    