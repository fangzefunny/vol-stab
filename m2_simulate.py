
import os 
import pickle
import argparse 
import numpy as np 
import pandas as pd
import multiprocessing as mp

from utils.model import subj
from utils.params import set_hyperparams
from utils.agents import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--agent_name', '-n', help='choose agent', default='RDModel2')
parser.add_argument('--data_set', '-d', help='choose data set', default='rew_con')
parser.add_argument('--mode', '-m', help='type of simulation', default='reg')
parser.add_argument('--group', '-g', help='choose agent', default='ind')
parser.add_argument('--seed', '-s', help='random seed', type=int, default=120)
parser.add_argument('--n_cores', '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=0)
args = parser.parse_args()

# create the folders for this folder
if not os.path.exists(f'{path}/simulations'):
    os.mkdir(f'{path}/simulations')
if not os.path.exists(f'{path}/simulations/{args.agent_name}'):
    os.mkdir(f'{path}/simulations/{args.agent_name}')

# define simulation fn
def simulate( data, sub_idx, args, seed):
    '''Simualte the data for a subject
    '''
    print( f'sim {sub_idx}')
    ## Define the RL model 
    model = subj( args.agent, seed)
    n_params = len( args.bnds)

    ## Obtain parameters 
    if args.mode == 'reg':
        if args.group == 'ind': 
            fname = f'{path}/fits/{args.agent_name}/params-{args.data_set}-{sub_idx}.csv'    
        elif args.group == 'avg':
            fname = f'{path}/fits/params-{args.data_set}-{args.agent_name}-avg.csv'      
        params = pd.read_csv( fname, index_col=0).iloc[0, 0:n_params].values
    elif args.mode == 'check':
        # in this simulation we replace the beta_stab with beta_vol
        if args.group == 'ind': 
            fname = f'{path}/fits/{args.agent_name}/params-{args.data_set}-{sub_idx}.csv'  
        elif args.group == 'avg':
            fname = f'{path}/fits/params-{args.data_set}-{args.agent_name}-avg.csv'
        df = pd.read_csv( fname, index_col=0)
        params = df.iloc[0, 0:n_params].values
        beta_stab_id = list(df.columns).index('β_stab')
        beta_vol_id  = list(df.columns).index('β_vol')
        params[beta_stab_id] = params[beta_vol_id]
    else:
        raise ValueError("Choose the correct simulation mode")

    ## synthesize the data and save
    return model.predict( {f'{sub_idx}':data}, params)

# define functions
def sim_paral( data, args):
    '''The simulate method used in Collins&Frank 2012

        "Twenty simulated experiments were each subject's
        individual squence and parameters， and then averaged 
        to represent subjects contribution."
    '''
    ## Get all the subject id 
    sub_ind = list(data.keys())

    ## Get the multiprocessing pool
    n_cores = args.n_cores if args.n_cores else int( mp.cpu_count()) 
    n_cores = np.min( [ n_cores, len(sub_ind)])
    pool = mp.Pool( n_cores)
    print( f'Using {n_cores} parallel CPU cores')

    ## Simulate data for each subject
    seed = args.seed 
    res = [ pool.apply_async( simulate, args=( data[sub_id], sub_id, args, seed+5*i))
                            for i, sub_id in enumerate(sub_ind)]
    for i, p in enumerate( res):
        sim_sample = p.get() 
        if i == 0:
            sim_data = sim_sample 
        else:
            sim_data = pd.concat( [ sim_data, sim_sample], axis=0, ignore_index=True)
    fname = f'{path}/simulations/{args.agent_name}/sim_{args.data_set}-mode={args.mode}.csv'
    sim_data.to_csv( fname, index = False, header=True)
    
if __name__ == '__main__':
    
    ## STEP 0: HYPERPARAMETER TUNING
    args = set_hyperparams(args)  

    ## STEP 1: LOAD DATA
    with open(f'{path}/data/{args.data_set}.pkl', 'rb') as handle:
        data = pickle.load( handle)

    ## STEP 2: SYNTHESIZE DATA
    sim_paral( data, args)