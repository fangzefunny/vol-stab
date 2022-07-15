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
path = os.path.dirname(os.path.abspath(__file__))

# -----  Preprocess to extract the control reward group ---- #

def get_pat_dict():

    ## Load data 
    data = pd.read_csv(f'{path}/data/participant_table_exp1.csv')
    pat_lst  = data['group']
    subj_lst = data['MID']

    ## Generate pat_dict
    pat_dict = dict()
    for i, subj in enumerate(subj_lst):
        dis = pat_lst[i]
        if (dis == 'control') or (dis == 'community'):
            dis = 'CON'
        pat_dict[subj] = dis
    
    return pat_dict

def remake_cols_idx(data, sub_id, dis_group, seed=42):
    '''Core preprocess fn
    '''
    # random generator
    rng = np.random.RandomState(seed)

    ## Replace some undesired col name 
    col_dict = { 'choice':       'humanAct',
                 'Unnamed: 0':   'trial',
                 'green_mag':    'mag0',
                 'blue_mag':     'mag1',
                 'block':        'b_type',
                 'green_outcome':'state'}
    data.rename(columns=col_dict, inplace=True)

    ## Change the trial index
    data['trial'] = data['trial'].apply(lambda x: x % 90)

    ## Change the action index
    # the raw data: left stim--1, right stim--0
    # I prefer:     left stim--0, right stim--1
    data['humanAct'] = data['humanAct'].fillna(rng.choice(2))
    data['humanAct'] = data['humanAct'].apply(lambda x: int(1-x))
    
    ## Change the state index
    # the raw data: left stim--1, right stim--0
    # I prefer:     left stim--0, right stim--1
    data['state'] = data['state'].apply(lambda x: int(1-x))

    ## Check if correct
    data['match'] = data.apply(lambda x: int(x['humanAct']==x['state']), axis=1) 
    
    ## Experiment type
    # stabe -- 'stab first' or 'vol first'
    data['exp_type'] = 'stab_first' if (data.loc[:89, 'b_type'].max() == 'stable'
                                ) else 'vol_first'

    ## True probability
    data.loc[:89, 'p_true'] = data.loc[:89, 'state'].mean()
    data.loc[90:, 'p_true'] = data.loc[90:, 'state'].mean()

    ## Give the label
    data['seq'] = data['p_true'].apply(lambda x: int(x >= .5))

    ## Add the sub id col
    data['sub_id'] = sub_id

    ## Add the patient group
    # MDD -- Depression
    # GAD -- Anxiety
    # CON -- control
    data['group'] = dis_group
    
    return data 

def preprocess(mode='rew_con'):
    '''Data prerprocessing  

        'rew_con': exp1+exp2 control group 
        'exp1_all': exp1 all patient
        'exp1_rew': exp1 rew 
    '''

    ## Create the processed data storage
    rew_con_data = dict()  # for fit 
    all_data     = []      # for visualize 

    ## Get the patient disease info
    pat_dict = get_pat_dict()

    ## Loop to preprocess each file
    # obtain all files under the exp1 list
    files = os.listdir(f'{path}/data/reward_con/exp1/')
    for file in files:
        # skip the folder 
        if not os.path.isdir(file): 
            # get the subid 
            sub_idx, rew_pain = file.split('_')[3], file.split('_')[4]  
            # get disease group
            try:
                dis = pat_dict[sub_idx]
            except:
                dis = 'CON'
            # decide the data extraction rule 
            if (mode == 'rew_con') or (mode=='exp1_rew'):
                cond = (dis=='CON') and (rew_pain=='rew')
            else:
                cond = (rew_pain=='rew')
            # get data for CON group subject     
            if cond:
                subj_data = remake_cols_idx(
                    pd.read_csv(f'{path}/data/reward_con/exp1/{file}'), sub_idx, dis
                )  
                rew_con_data[sub_idx] = {'0': subj_data}
                all_data.append(subj_data) 

    n_exp1 = len(rew_con_data.keys())

    ## Continue if mode is rew_con
    if mode != 'rew_con':
        print(f'#subj in {mode}: {n_exp1}')
        with open(f'{path}/data/{mode}.pkl', 'wb')as handle:
            pickle.dump(rew_con_data, handle)

        ## Save the preprocessed for visualize
        all_data = pd.concat(all_data, axis=0)
        fname    = f'{path}/data/{mode}.csv'
        all_data.to_csv(fname)

    else: 
        # obtain all files under the exp2 list
        files = os.listdir(f'{path}/data/reward_con/exp2/')
        for file in files:
            # skip the folder 
            if not os.path.isdir(file): 
                # get the subid 
                rew_pain, sub_idx = file.split('_')[3], file.split('_')[4]  
                # get disease group
                dis = 'CON'
                # get data for CON group subject :
                subj_data = remake_cols_idx(
                    pd.read_csv(f'{path}/data/reward_con/exp2/{file}'), sub_idx, dis
                )  
                rew_con_data[sub_idx] = {'0': subj_data} 
                all_data.append(subj_data) 
        n_exp2 = len(rew_con_data.keys()) - n_exp1
            
        ## Save the preprocessed for fit 
        with open(f'{path}/data/{mode}.pkl', 'wb')as handle:
            pickle.dump(rew_con_data, handle)
        print(f'#subj in exp1: {n_exp1}\n#subj in exp2: {n_exp2}')

        ## Save the preprocessed for visualize
        all_data = pd.concat(all_data, axis=0)
        fname    = f'{path}/data/{mode}.csv'
        all_data.to_csv(fname, index = False, header=True)

def get_rdcurves(data_set):
    
    # find is patient
    with open(f'{path}/data/{data_set}.pkl', 'rb') as handle:
        data = pickle.load(handle)

    # index the data 
    mag0_lst = data['n26']['mag0'][data['n26']['b_type']==1].values
    mag1_lst = data['n26']['mag1'][data['n26']['b_type']==1].values
        
    Stab = np.array([[.75, .25]]).T
    Vol  = np.array([[ .5,  .5]]).T
    betas = np.logspace(np.log10(.1), np.log10(10), 50)

    # stable rd curves:
    conds = ['Stab', 'Vol'] 
    rdcurves = dict()
    for cond in conds:

        # create placeholder 
        R_theo = np.ones([len(mag0_lst), len(betas),]) + np.nan
        V_theo = np.ones([len(mag0_lst), len(betas),]) + np.nan

        # get p_s
        p_s = eval(cond)

        for t in range(len(mag0_lst)):

            Q = np.array([[mag0_lst[t],           0],
                          [           0, mag1_lst[t]]])

            for betai, beta in enumerate(betas):

                # get the optimal channel for each tradeoff
                pi_a1s, p_a = blahut_arimoto(-Q, p_s,
                                              beta)
                # calculate the expected distort (-utility)
                # EU = ∑_s,a p(s)π(a|s)Q(s,a)
                theo_util  = np.sum(p_s * pi_a1s * Q)
                # Rate = β*EU - ∑_s p(s) Z(s) 
                # Z(s) = log ∑_a p(a)exp(βQ(s,a))  # nSx1
                Zstate     = logsumexp(beta * Q + np.log(p_a.T), 
                                    axis=-1, keepdims=True)
                theo_rate  = beta * theo_util - np.sum(p_s * Zstate)

                # record
                R_theo[t, betai]  = theo_rate
                V_theo[t, betai]  = theo_util

        rdcurves[cond] = [R_theo, V_theo]
    
    with open(f'{path}/data/rdcurves-{data_set}.pkl', 'wb')as handle:
        pickle.dump(rdcurves, handle)

if __name__ == '__main__':

    ## STEP0: PREPROCESS EXP1 DATA 
    modes = ['rew_con', 'exp1_all', 'exp1_rew']
    for m in modes:
        preprocess(m) 

