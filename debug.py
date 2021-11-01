import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings

from scipy.special import logsumexp 
from utils.brains import *
from utils.model import subj
from m0_preprocess import remake_cols_idx


def sim( brain, sub_idx, seed=2014, data_set='rew_data_exp1'):

    # fix random seed
    np.random.seed( seed)

    # get the data 
    dname = f'data/data_raw_exp1/behavioral_trial_table_{sub_idx}_rew_modelready.csv'
    task = { f'{sub_idx}': remake_cols_idx(pd.read_csv( dname), sub_idx, '')}

    # define the RL2 model 
    model = subj( eval(brain))

    # if there is no input parameter,
    # choose the best model,
    # the last column is the loss which is ignored
    fname = f'results/params-{data_set}-{brain}-avg.csv'
    params = pd.read_csv( fname, index_col=0).iloc[0, 0:-1].values
    print(pd.read_csv( fname, index_col=0).iloc[0, :])
    
    # synthesize the data and save
    return model.pred( params, task)    


def plot_sim( sim_data):

    # get the trials
    trials  = sim_data['trial'].values
    outcome = sim_data['state'].values
    NLLs    = sim_data['nll'].values
    tot_NLL = sim_data['nll'].values.sum()
    b1_NLL = sim_data['nll'].values[:90].sum()
    b2_NLL = sim_data['nll'].values[90:].sum()
    
    # get the true prob 
    p1 = np.sum(sim_data.loc[ :90, 'state'] == 0) / 90 
    p2 = np.sum(sim_data.loc[ 90:, 'state'] == 0) / 90 
    if sim_data['b_type'][0]:
        p1 = [p1] * 90
        if p2 > .5:
            p2 = [.8]*20+[.2]*20+[.8]*20+[.2]*20+[.8]*10
        else:
            p2 = [.2]*20+[.8]*20+[.2]*20+[.8]*20+[.2]*10
    else:
        p2 = [p2] * 90
        if p1 > .5:
            p1 = [.8]*20+[.2]*20+[.8]*20+[.2]*20+[.8]*10
        else:
            p1 = [.2]*20+[.8]*20+[.2]*20+[.8]*20+[.2]*10
    
    # get the human response 
    p         = sim_data['p_s'] 
    human_act = -1 * ((sim_data['human_act'].values == 1) * 2 - 1)
    sim_act   = -1 * ((sim_data['action'].values == 1) * 2 - 1)
    prob_act  = sim_data['act_acc'].values * sim_act
    human_a1  = sim_data['human_act'].values == 0
    avg_a1    = [np.sum(human_a1[:90]) / 90] * 90 + [np.sum(human_a1[90:]) / 90] * 90

    my_dpi = 150
    plt.figure( figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

    plt.subplot( 3, 1, 1)
    plt.plot( trials, p1 + p2, color=[.6, .6, .6])
    plt.plot( trials, p, color='r')
    plt.title( f'NLL: {tot_NLL:.3f}; block 1: {b1_NLL:.3f}; block 2: {b2_NLL:.3f}')
    
    plt.subplot( 3, 1, 2)
    plt.bar( trials, human_act, color=[.6, .6, .6])
    plt.plot( trials, prob_act, color='r')

    plt.subplot( 3, 1, 3)
    plt.plot( trials, NLLs, color=[.6, .6, .6])
    plt.ylim( [ 0, 2.5])
    # plt.scatter( trials, human_a1, color='b')
    # plt.plot( trials, avg_a1, color='b')


if __name__ == '__main__':

    # select the sub idx
    sub_id = 'n25'
    
    # do simulation 
    plot_sim(sim( 'max_mag', sub_id, seed=1234))

