# import pkgs 
import os
import pickle
import numpy as np 
import pandas as pd
from scipy.special import logsumexp
from utils.analyze import blahut_arimoto

'''
What we need to do

1. Add exp2
2. Add bifactor
'''

# path to the current file 
path = os.path.dirname( os.path.abspath(__file__))

def get_pat_dict( data_set):

    ## Load data 
    data = pd.read_csv( f'{path}/data/participant_table_{data_set}.csv')
    pat_lst  = data['group']
    subj_lst = data['MID']

    ## Generate pat_dict
    pat_dict = dict()
    for i, subj in enumerate(subj_lst):
        dis = pat_lst[i]
        if ( dis == 'control') or ( dis == 'community'):
            dis = 'CON'
        pat_dict[ subj] = dis
    
    return pat_dict

def remake_cols_idx( data, sub_id, dis_group):
    '''Core preprocess fn
    '''
    ## Replace some undesired col name 
    col_dict = { 'choice':       'action',
                 'Unnamed: 0':   'trial',
                 'green_mag':    'mag0',
                 'blue_mag':     'mag1',
                 'block':        'b_type',
                 'green_outcome':'state'}
    data.rename( columns=col_dict, inplace=True)

    ## Change the action index
    # the raw data: left stim--1, right stim--0
    # I prefer:     left stim--0, right stim--1
    data['action'] = 1 - data['action']
    data.loc[data['action'].isna(), 'action']= int(np.random.choice([0,1]))

    ## Change the state index
    # the raw data: left stim--1, right stim--0
    # I prefer:     left stim--0, right stim--1
    data['state'] = 1 - data['state']

    ## Numeric the stab-vol index 
    # stable--1, volatile--0
    data['b_type'] = (data['b_type'] == 'stable') * 1

    ## Add the sub id col
    data['sub_id'] = sub_id

    ## Add the patient group
    # MDD -- Depression
    # GAD -- Anxiety
    # CON -- control
    data['group'] = dis_group
    
    return data 

def preprocess( exp_folder, exp_id):

    ## Create the processed data storage
    rew_data = dict()
    pain_data = dict()

    ## Get the patient disease info
    pat_dict = get_pat_dict( exp_id)

    ## Loop to preprocess each file
    # obtain all files under the exp1 list
    files = os.listdir( f'{path}/data/{exp_folder}/')
    for file in files:
        # skip the folder 
        if not os.path.isdir( file): 
            # get the subid 
            sub_idx, rew_pain = file.split('_')[3], file.split('_')[4]  
            # get disease group
            try:
                dis = pat_dict[sub_idx]
            except:
                dis = 'CON'
            # get data for each subject
            eval(f'{rew_pain}_data')[ sub_idx] = remake_cols_idx(
                pd.read_csv(f'{path}/data/{exp_folder}/{file}'), sub_idx, dis
            )  
            
    ## Save the preprocessed 
    with open( f'{path}/data/rew_data_{exp_id}.pkl', 'wb')as handle:
        pickle.dump( rew_data, handle)
    with open( f'{path}/data/pain_data_{exp_id}.pkl', 'wb')as handle:
        pickle.dump( pain_data, handle)

def get_rdcurves( data_set):
    
    # find is patient
    with open(f'{path}/data/{data_set}.pkl', 'rb') as handle:
        data = pickle.load( handle)

    # index the data 
    mag0_lst = data['n26']['mag0'][ data['n26']['b_type']==1].values
    mag1_lst = data['n26']['mag1'][ data['n26']['b_type']==1].values
        
    Stab = np.array( [[ .75, .25]]).T
    Vol  = np.array( [[  .5,  .5]]).T
    betas = np.logspace( np.log10(.1), np.log10(10), 50)

    # stable rd curves:
    conds = [ 'Stab', 'Vol'] 
    rdcurves = dict()
    for cond in conds:

        # create placeholder 
        R_theo = np.ones( [ len(mag0_lst), len(betas),]) + np.nan
        V_theo = np.ones( [ len(mag0_lst), len(betas),]) + np.nan

        # get p_s
        p_s = eval(cond)

        for t in range( len(mag0_lst)):

            Q = np.array([[ mag0_lst[t],           0],
                          [            0, mag1_lst[t]]])

            for betai, beta in enumerate(betas):

                # get the optimal channel for each tradeoff
                pi_a1s, p_a = blahut_arimoto( -Q, p_s,
                                              beta)
                # calculate the expected distort (-utility)
                # EU = ∑_s,a p(s)π(a|s)Q(s,a)
                theo_util  = np.sum( p_s * pi_a1s * Q)
                # Rate = β*EU - ∑_s p(s) Z(s) 
                # Z(s) = log ∑_a p(a)exp(βQ(s,a))  # nSx1
                Zstate     = logsumexp( beta * Q + np.log(p_a.T), 
                                    axis=-1, keepdims=True)
                theo_rate  = beta * theo_util - np.sum( p_s * Zstate)

                # record
                R_theo[t, betai]  = theo_rate
                V_theo[t, betai]  = theo_util

        rdcurves[cond] = [ R_theo, V_theo]
    
    with open( f'{path}/data/rdcurves-{data_set}.pkl', 'wb')as handle:
        pickle.dump( rdcurves, handle)

if __name__ == '__main__':

    ## STEP0: PREPROCESS EXP1 DATA 
    preprocess( 'data_raw_exp1', 'exp1')    

    ## STEP1: CALCULATE RD CURVES   
    get_rdcurves( 'rew_data_exp1') 


