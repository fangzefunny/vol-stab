import os 
import pickle
import numpy as np
import pandas as pd 

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

def get_bic( data_set, model, sub_idx, m):
    '''Calculate the bic of the model 
    '''
    fname = f'{path}/results/params-{data_set}-{model}-{sub_idx}.csv'
    res   = pd.read_csv(fname)
    model_nll = res.iloc[0, -1]
    n_params  = res.shape[1] - 2
    return 2 * model_nll + np.log(m) * n_params

def bic_table( model_lst, data_set):
    '''Generate bic for each model and subj
    '''
    ## Load data and subj lst 
    with open(f'{path}/data/{data_set}.pkl', 'rb') as handle:
        data = pickle.load( handle)
    subj_lst = data.keys()

    ## Create table
    tab = np.zeros( [ len(subj_lst), len(model_lst)]) + np.nan
    for i, subj in enumerate( subj_lst):
        sub_data = data[subj]
        m        = sub_data.shape[0]
        for j, model in enumerate( model_lst):
            tab[ i, j] = get_bic( data_set, model, subj, m)
        
    ## save table
    tab = pd.DataFrame( tab, columns=model_lst, index=subj_lst)
    tname = f'{path}/tables/bic_table-{data_set}.csv'
    try:
        tab.to_csv( tname)
    except:
        os.mkdir( f'{path}/tables')
        tab.to_csv( tname)

def get_criter( model_lst, data_set):
    '''Num of the best model in the subjects
    '''
    ## Load the bic table 
    tab = pd.read_csv( f'{path}/tables/bic_table-{data_set}.csv')
    tab = tab.iloc[:,1:].to_numpy()

    ## max of each row 
    best_num = np.sum( np.tile( np.min( tab, axis=1), [len(model_lst), 1]).T == tab, axis=0)
    best_num = pd.DataFrame( best_num.reshape([1, -1]), columns=model_lst)
    best_num.to_csv( f'{path}/tables/criter1-{data_set}.csv')

    ## sum of each row
    sum_bic = np.sum( tab, axis=0)
    sum_bic = pd.DataFrame( sum_bic.reshape([1, -1]), columns=model_lst)
    sum_bic.to_csv( f'{path}/tables/criter2-{data_set}.csv')

    ## sum of each row


if __name__ == '__main__':

    ## STEP0: GET THE TARGET MODEL AND DATA SETS 
    model_lst = [ 'model1', 'model2', 'model7', 
                  'model8', 'model11', 
                  'modelE1', 'modelE2', 'RRmodel']
    data_sets = [ 'rew_data_exp1']

    ## STEP1: GENERATE TABLES WE PREFERRED
    for data_set in data_sets:
        
        # get BIC table
        bic_table( model_lst, data_set)

        # get two comparison criteria
        get_criter( model_lst, data_set)
