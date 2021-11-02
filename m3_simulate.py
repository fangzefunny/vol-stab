import argparse 
import os 
import pickle
import pandas as pd
from utils.model import subj
from utils.hyperparams import set_hyperparams
from utils.brains import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--brain_name', '-n', help='choose agent', default='RRmodel')
parser.add_argument('--data_set', '-d', help='choose data set', default='rew_data_exp1')
parser.add_argument('--fit_mode', '-m', help='fitting methods', type = str, default='mle')
args = parser.parse_args()

# define functions
def sim_subj( train_data, sub_idx, args, params=[]):

    # define the RL2 model 
    model = subj( args.brain)

    # if there is no input parameter,
    # choose the best model,
    # the last column is the loss which is ignored
    if len(params) == 0:
        fname = f'{path}/results/params-{args.data_set}-{args.brain_name}-{sub_idx}.csv'
        params = pd.read_csv( fname, index_col=0).iloc[0, 0:-1].values
    
    # synthesize the data and save
    sim_data = model.pred( params, train_data)

    return sim_data

def simulate( args):

    ## Load data 
    with open( f'{path}/data/{args.data_set}.pkl', 'rb')as handle:
        train_data = pickle.load( handle)
    
    ## Simulate data placeholder
    sim_data = dict() 

    ## Start simulation
    for sub_idx in train_data.keys():
        # get subject data
        sub_data = {sub_idx:train_data[ sub_idx]}
        # simulate
        sim_data[sub_idx] = sim_subj( sub_data, sub_idx, args)

    ## Save the simulated data  
    with open( f'{path}/data/sim-{args.data_set}-{args.brain_name}.pkl', 'wb')as handle:
        pickle.dump( sim_data, handle)    

if __name__ == '__main__':
    
    ## STEP0: HYPERPARAMETER TUNING
    args = set_hyperparams(args)  

    ## STEP1: PREDICT/SIMULATE
    simulate( args)