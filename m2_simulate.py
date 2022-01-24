
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
parser.add_argument('--n_sim', '-f', help='f simulations', type=int, default=1)
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
def simulate( data, args, seed, in_params=[]):

    # define the RL model 
    model = subj( args.agent, seed)
    n_params = len( args.bnds)

    ## Loop to choose the best model for simulation
    for i, sub_idx in enumerate( data.keys()): 

        if len( in_params) ==0:
            if args.group == 'ind': 
                try:
                    fname = f'{path}/fits/{args.agent_name}/params-rew_con-{sub_idx}.csv'  
                except:
                    fname = f'{path}/fits/{args.agent_name}/params-rew_data_exp1-{sub_idx}.csv'    
            elif args.group == 'avg':
                fname = f'{path}/fits/params-{args.data_set}-{args.agent_name}-avg.csv'      
            params = pd.read_csv( fname, index_col=0).iloc[0, 0:n_params].values
        else:
            params = in_params

        # synthesize the data and save
        seed += 1
        sim_sample = model.predict( {f'{sub_idx}':data[ sub_idx]}, params)
        if i == 0:
            sim_data = sim_sample 
        else:
            sim_data = pd.concat( [ sim_data, sim_sample], axis=0, ignore_index=True)

    return sim_data

# define functions
def sim_paral( data, args):
    '''The simulate method used in Collins&Frank 2012

        "Twenty simulated experiments were each subject's
        individual squence and parameters， and then averaged 
        to represent subjects contribution."
    '''
    ## Get the multiprocessing pool
    n_cores = args.n_cores if args.n_cores else int( mp.cpu_count()) 
    n_cores = np.min( [ n_cores, args.n_sim])
    pool = mp.Pool( n_cores)
    print( f'Using {n_cores} parallel CPU cores')

    ## Simulate data for n_sim times 
    seed = args.seed 
    res = [ pool.apply_async( simulate, args=( data, args, seed+5*i))
                            for i in range( args.n_sim)]
    for i, p in enumerate( res):
        sim_data = p.get() 
        fname = f'{path}/simulations/{args.agent_name}/sim_{args.data_set}-idx{i}.csv'
        sim_data.to_csv( fname, index = False, header=True)
    

if __name__ == '__main__':
    
    ## STEP 0: HYPERPARAMETER TUNING
    args = set_hyperparams(args)  

    ## STEP 1: LOAD DATA
    with open(f'{path}/data/{args.data_set}.pkl', 'rb') as handle:
        data = pickle.load( handle)

    ## STEP 2: SYNTHESIZE DATA
    sim_paral( data, args)