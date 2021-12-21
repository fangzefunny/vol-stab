import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings

from scipy.special import logsumexp 
from utils.agents import *
from utils.model import subj
from m0_preprocess import remake_cols_idx
from scipy.optimize import minimize

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

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
    return model.predict( params, task)    

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

def eligbility_trace( sub_idx, seed=42):

    ## Load data as environment
    dname = f'data/data_raw_exp1/behavioral_trial_table_{sub_idx}_rew_modelready.csv'
    data = remake_cols_idx(pd.read_csv( dname), sub_idx, '')
    p1 = np.sum(data.loc[ :90, 'state'] == 0) / 90 
    p2 = np.sum(data.loc[ 90:, 'state'] == 0) / 90 
    if data['b_type'][0]:
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
    p = p1 + p2 

    trials = data['trial'].values
    t1 = trials[:90]
    t2 = trials[90:]
    states = data['state'].values
    nS = 2 
    prev = 0#np.ones([ nS, 1]) / nS
    beta = .99
    alpha = .2
    alpha_q = .2
    e_p1 = [] 
    q_p1 = []
    ps = np.ones([ nS, 1]) / nS 
    qs = np.ones([ nS, 1]) / nS 
    # for t in t1:
    #     e_p1.append( ps[ 0, 0])
    #     q_p1.append( qs[ 0, 0])
    #     state = states[t]
    #     I_st = np.zeros( [ nS, 1])
    #     I_st[ state, 0] = 1 
    #     #update
    #     prev = beta * prev + (1-beta) * (I_st)
    #     ps += alpha * (prev - ps)
    #     #update q
    #     qs += alpha * (I_st - qs)
    for t in trials:
        e_p1.append( ps[ 0, 0])
        q_p1.append( qs[ 0, 0])
        state = states[t]
        I_st = np.zeros( [ nS, 1])
        I_st[ state, 0] = 1 
        #update
        prev = beta * prev + I_st
        prev /= prev.sum()
        ps += alpha * (prev - ps)
        #update q
        qs += alpha_q * (I_st - qs) 
    
    e_p2 = [] 
    q_p2 = []
    ps = np.ones([ nS, 1]) / nS 
    qs = np.ones([ nS, 1]) / nS 
    for t in t2:
        e_p2.append( ps[ 0, 0])
        q_p2.append( qs[ 0, 0])
        state = states[t]
        I_st = np.zeros( [ nS, 1])
        I_st[ state, 0] = 1 
        #update
        prev = beta * prev +  (I_st)
        prev /= prev.sum()
        ps += alpha * (prev - ps)
        #update q
        qs += alpha_q * (I_st - qs)

    fig, ax = plt.subplots( 2, 1, figsize=(4.5,4.5))
    ax[0].plot( trials, p1 + p2, color=[ .7,.7,.7])
    ax[0].plot( trials, e_p1, color='r') 
    ax[0].plot( trials, q_p1, color='b') 
    ax[1].plot( t2, p2, color=[ .7,.7,.7])
    ax[1].plot( t2, e_p2, color='r') 
    ax[1].plot( t2, q_p2, color='b') 

def sim_e( param, sub_idx, seed=42):

    # get param 
    alpha_s, alpha_v = param  

    # rng
    rng = np.random.RandomState( seed)

    ## Load data as environment
    dname = f'data/data_raw_exp1/behavioral_trial_table_{sub_idx}_rew_modelready.csv'
    data = remake_cols_idx(pd.read_csv( dname), sub_idx, '')
    trials  = data['trial'].values
    states  = data['state'].values
    mag0s   = data['mag0'].values
    mag1s   = data['mag1'].values
    b_types = data['b_type'].values
    # p1 = np.sum(data.loc[ :90, 'state'] == 0) / 90 
    # p2 = np.sum(data.loc[ 90:, 'state'] == 0) / 90 
    # if data['b_type'][0]:
    #     p1 = [p1] * 90
    #     if p2 > .5:
    #         p2 = [.8]*20+[.2]*20+[.8]*20+[.2]*20+[.8]*10
    #     else:
    #         p2 = [.2]*20+[.8]*20+[.2]*20+[.8]*20+[.2]*10
    # else:
    #     p2 = [p2] * 90
    #     if p1 > .5:
    #         p1 = [.8]*20+[.2]*20+[.8]*20+[.2]*20+[.8]*10
    #     else:
    #         p1 = [.2]*20+[.8]*20+[.2]*20+[.8]*20+[.2]*10
    # pis = p1+p2

    # init model
    nS = 2
    p_s = np.ones( [ nS, 1]) / nS 
    prev = np.ones( [ nS, 1]) / nS 
    lamb1  = .99
    alpha1 = .3
    beta  = 5
    q_s = np.ones( [ nS, 1]) / nS
    nll = 0 
    for t in trials:
        # get state and mag 
        state = states[t]
        mag0  = mag0s[t]
        mag1  = mag1s[t]
        b_type = b_types[t]  
        I_st  = np.zeros( [ nS, 1])
        I_st[ state, 0] = 1

        # choose by model e
        prev = lamb1*prev + I_st
        prev /= prev.sum()
        p_s += alpha1 * ( prev - p_s)
        v = p_s[ 0, 0] * mag0 - p_s[ 1, 0] * mag1 
        p = 1 / ( 1 + np.exp(-beta * v)) 
        act = rng.choice( [0,1], p=[p, 1-p])

        ## fit model lr 
        alpha = alpha_s if b_type else alpha_v 
        q_s += alpha * ( I_st - q_s)
        v = q_s[ 0, 0] * mag0 - q_s[ 1, 0] * mag1 
        p = 1 / ( 1 + np.exp(-beta * v)) 
        p_q = [ p, 1 - p]
        nll += -np.log( p_q[ act] + eps_) 
    
    return nll 


def check( sub_idx, n_fit =20, seed=42):
    rng = np.random.RandomState(seed) 
    min_val = np.inf
    for i in range(n_fit):
        param0 = [ rng.rand() for _ in range(2)] 
        res = minimize( sim_e, param0, 
                        args=(sub_idx, seed+i),
                        bounds = [(0,1), (0,1)])
        if res.fun < min_val:
            print( res.fun, res.x )
            min_val = res.fun
            param_opt = res.x 
    return param_opt


if __name__ == '__main__':

    # select the sub idx
    sub_id = 'n32'
    
    # do simulation 
    #plot_sim(sim( 'RRmodel', sub_id, seed=1234))
    #eligbility_trace( sub_id)
    #plt.show()
    print(check( sub_id))

