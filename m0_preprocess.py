# import pkgs 
import os
import pickle
import numpy as np 
import pandas as pd

'''
What we need to do

1. Understand what is "run"
2. Add exp2
3. Add bifactor
'''

# path to the current file 
path = os.path.dirname( os.path.abspath(__file__))

def remake_cols_idx( data, sub_id):
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
    # I prefer: left stim--0, right stim--1
    data['action'] = 1 - data['action']
    data[data['action'].isna()] = int(np.random.choice([0,1]))

    ## Numeric the stab-vol index 
    # stable--1, volatile--0
    data['b_type'] = (data['b_type'] == 'stable') * 1

    ## Add the sub id col
    data['sub_id'] = sub_id
    
    return data 

def preprocess( exp_folder, exp_id):

    ## Create the processed data storage
    processed_data = dict()

    ## Loop to preprocess each file
    # obtain all files under the exp1 list
    files = os.listdir( f'{path}/data/{exp_folder}/')
    for file in files:
        # skip the folder 
        if not os.path.isdir( file): 
            # get the subid 
            sub_id = file.split('_')[3]  
            # get data for each subject
            processed_data[ sub_id] = remake_cols_idx(
                pd.read_csv(f'{path}/data/{exp_folder}/{file}'), sub_id
            ) 
    
    ## Save the preprocessed 
    with open( f'{path}/data/processed_{exp_id}.pkl', 'wb')as handle:
        pickle.dump( processed_data, handle)

if __name__ == '__main__':

    # preprocess exp1 
    preprocess( 'data_raw_exp1', 'exp1')        


    
    
    
    
    


    
    
    
