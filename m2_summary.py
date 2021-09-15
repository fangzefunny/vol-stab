import os 
import pickle
from utils.model import subj 
import numpy as np
import pandas as pd 

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

def get_bic( model, sub_idx, m):
    '''Calculate the bic of the model 
    '''
    fname = f'{path}/results/params-{model}-{sub_idx}.csv'
    res   = pd.read_csv(fname)
    model_nll = res.iloc[0, -1]
    n_params  = res.shape[1] - 2
    return 2 * model_nll + np.log(m) * n_params

def bic_table( model_lst, data_set):
    '''Generate bic for each model and subj
    '''
    ## Load data and subj lst 
    with open(f'{path}/data/processed_{data_set}.pkl', 'rb') as handle:
        data = pickle.load( handle)
    subj_lst = data.keys()

    ## Create table
    tab = np.zeros( [ len(subj_lst), len(model_lst)]) + np.nan
    for i, subj in enumerate( subj_lst):
        sub_data = data[subj]
        m        = sub_data.shape[0]
        for j, model in enumerate( model_lst):
            tab[ i, j] = get_bic( model, subj, m)
        
    ## save table
    tname = f'{path}/tables/bic_table.csv'
    try:
        tname.to_csv( tname)
    except:
        os.mkdir( f'{path}/tables')
        tname.to_csv( tname)

def get_criter_1( model_lst):
    '''Num of the best model in the subjects
    '''




if __name__ == '__main__':

    ## create compare list
    model_lst = [ 'model1', 'model2', 'model7', 
                  'model8', 'model11', 'RRmodel']

    ## Get BIC table
    bic_table( model_lst, 'exp1')

    ## Criterion 1:
    #get_criter_1()
