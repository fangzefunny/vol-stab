# import pkgs 
import os
import pickle
import numpy as np 
import pandas as pd

# path to the current file 
path = os.path.dirname(os.path.abspath(__file__))

def get_feedback_subid(fname, exp_id):
    d = {'rew':  'gain',
         'pain': 'loss',
         'gain': 'gain',
         'loss': 'loss'}
    if exp_id == 'exp1':
        sub_id        = fname.split('_')[3]
        feedback_type = fname.split('_')[4]
    elif exp_id == 'exp2':
        sub_id        = fname.split('_')[4]
        feedback_type = fname.split('_')[3]
    return sub_id, d[feedback_type]

def remake_cols_idx(data, sub_id, feedback_type, exp_id, seed=42):
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

    ## Change the action index
    # the raw data: left stim--1, right stim--0
    # I prefer:     left stim--0, right stim--1
    data['humanAct'] = data['humanAct'].fillna(rng.choice(2))
    data['humanAct'] = data['humanAct'].apply(lambda x: int(1-x))
    
    ## Change the state index
    # the raw data: left stim--1, right stim--0
    # I prefer:     left stim--0, right stim--1
    data['state'] = data['state'].apply(lambda x: int(1-x))

    ## Change the block type index 
    data['b_type'] = data['b_type'].apply(lambda x: x[:3])

    ## Check if correct
    data['match'] = data.apply(lambda x: int(x['humanAct']==x['state']), axis=1) 
    
    ## Add the sub id col
    data['sub_id'] = sub_id

    ## Add the feedback type 
    data['feedback_type'] = feedback_type

    ## Add which experiment id 
    data['exp_id'] = exp_id

    return data 

def get_subinfo():
    exp_id = 'exp1'
    d1 = pd.read_csv(f'{path}/data/participant_table_{exp_id}.csv')[
                        ['MID', 'group_just_patients', 
                        'STAI_Trait_anx', 'STAI_Trait_dep']]
    d1 = d1.rename(columns={'group_just_patients': 'group',
                            'STAI_Trait_anx': 'anx_lvl',
                            'STAI_Trait_dep': 'dep_lvl'})
    d1['group'] = d1['group'].fillna('HC')

    exp_id = 'exp2'
    d2 = pd.read_csv(f'{path}/data/participant_table_{exp_id}.csv')[['MID']]
    d2['group'] = 'HC'

    sub_info = pd.concat([d1, d2], axis=0)
    sub_info = sub_info.rename(columns={'MID': 'sub_id'})

    # get the group
    sub_info1 = sub_info.groupby(by=['sub_id'])['group'].apply('-'.join).reset_index()
    sub_info1['group'] = sub_info1['group'].apply(lambda x: x.split('-')[0])
    # get the syndrome
    sub_info2 = sub_info.groupby(by=['sub_id']).mean().reset_index()
    # paste them  up 
    sub_info = sub_info1.join(sub_info2.set_index('sub_id'), 
                        on='sub_id', how='left')

    return sub_info

def preprocess(exp=['exp1', 'exp2']):

    for_analyze = []

    for exp_id in exp:
        
        # all files under the folder
        files = os.listdir(f'{path}/data/data_raw_{exp_id}')

        for file in files:
            
            # get sub_id and feedback_type
            sub_id, feedback_type = get_feedback_subid(file, exp_id)
            
            # remake some columns 
            fname = f'{path}/data/data_raw_{exp_id}/{file}'
            block_data = remake_cols_idx(pd.read_csv(fname),
            sub_id=sub_id, feedback_type=feedback_type, exp_id=exp_id)

            # append into storages
            for_analyze.append(block_data)

    # append into a large dataframe 
    for_analyze = pd.concat(for_analyze, axis=0)

    # get the subject information 
    sub_info = get_subinfo()

    # join two dataframe on key 'sub_id'
    for_analyze = for_analyze.join(sub_info.set_index('sub_id'), 
                        on='sub_id', how='left')

    # save for analyze
    idx = 'all' if (len(exp) == 2) else exp[0]
    fname = f'{path}/data/{idx}_data.csv'
    for_analyze.to_csv(fname, index = False, header=True)

    return for_analyze

def split_data(data, mode):

    # create storage
    for_fit = {}

    # split the data for fit
    sub_Lst = data['sub_id'].unique()
    exp_Lst = data['exp_id'].unique()
    idx = '' if (len(exp_Lst) == 2) else exp_Lst[0]

    for sub_id in sub_Lst:

        for_fit[sub_id] = {}
        condi = f'sub_id=="{sub_id}" & feedback_type=="{mode}"'
        block_data = data.query(condi)
        if block_data.empty is not True:
            for_fit[sub_id] = {0: block_data.reset_index()}
        else:
            for_fit.pop(f'{sub_id}')

    # save for fit 
    with open(f'{path}/data/{mode}_{idx}data.pkl', 'wb')as handle:
        pickle.dump(for_fit, handle)


if __name__ == '__main__':

    data = preprocess(['exp1'])
    split_data(data, mode='gain')
    split_data(data, mode='loss')
