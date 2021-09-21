import os
import pickle 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 



# find the current path
path = os.path.dirname(os.path.abspath(__file__))
dpi  = 500


def split_data( data_set, model):

    ## Get conditions 
    group_lst = [ 'MDD', 'GAD', 'CON']
    block_lst = [ 1, 0] #
    block_nm  = [ 'Stab', 'Vol']

    ## Prepare the dict for visualization
    EQ_dict = dict()
    pi_comp_dict = dict()
    for group in group_lst:
        for block in block_nm:
            EQ_dict[f'{group}-{block}'] = list()
            pi_comp_dict[f'{group}-{block}'] = list()

    ## Load simulate data 
    with open(f'{path}/data/sim-{data_set}-{model}.pkl', 'rb') as handle:
        sim_data = pickle.load( handle)

    sample_lst = list(sim_data.keys())
    sample_lst.pop( sample_lst.index('n29'))

    ## Loop to store data for plot
    for sub_idx in sample_lst:
        # create key name
        sub_data = sim_data[sub_idx]
        kname = sub_data['group'][0]
        for is_stab, nm in zip( block_lst, block_nm):
            ind = (sub_data['b_type'] == is_stab)
            EQ_dict[ f'{kname}-{nm}'].append( np.mean( sub_data['EQ'][ind].values))
            pi_comp_dict[ f'{kname}-{nm}'].append( 
                    np.clip( np.mean( sub_data['pi_comp'][ind].values), 0, np.log(2)))
    
    return EQ_dict, pi_comp_dict



def vis( rdcurves, foi_dict, title_str='', theta=0):

    EQ_dict, pi_comp_dict = foi_dict 

    Red     = np.array([ 255, 118, 117]) / 255
    Blue    = np.array([   9, 132, 227]) / 255
    conds   = [ 'Stab', 'Vol']
    colors  = [ Red, Blue]
    nr = 2
    nc = 2 
    fig, axes = plt.subplots( nr, nc, figsize=(4*nr, 4*nr))
    plt.rcParams.update({'font.size': 15})
    
    # general figures 
    ax = axes[ 0, 0]
    for cond, cl in zip(conds,colors):
        ax.plot( np.mean(rdcurves[cond][0],0), np.mean(rdcurves[cond][1],0), color=cl, linewidth=3)
    ax.scatter( pi_comp_dict['MDD-Stab'], EQ_dict['MDD-Stab'], color=Red,  marker='x')
    ax.scatter( pi_comp_dict[ 'MDD-Vol'], EQ_dict[ 'MDD-Vol'], color=Blue, marker='x')
    ax.scatter( pi_comp_dict['GAD-Stab'], EQ_dict['GAD-Stab'], color=Red,  marker='D')
    ax.scatter( pi_comp_dict[ 'GAD-Vol'], EQ_dict[ 'GAD-Vol'], color=Blue, marker='D')
    ax.scatter( pi_comp_dict['CON-Stab'], EQ_dict['CON-Stab'], color=Red,  marker='o')
    ax.scatter( pi_comp_dict[ 'CON-Vol'], EQ_dict[ 'CON-Vol'], color=Blue, marker='o')
    ax.set_title( f'Reward-complexity ({title_str})')
    ax.set_xticks([])
    ax.set_ylim( [ .2, .6])

    # show legend 
    ax = axes[ 0, 1]
    sz=20
    for cond, cl in zip(conds,colors):
        ax.plot( np.nan, np.nan, color=cl, linewidth=3)
    ax.plot( np.nan, np.nan, color=Red,  marker='x', linestyle='None')
    ax.plot( np.nan, np.nan, color=Blue, marker='x', linestyle='None')
    ax.plot( np.nan, np.nan, color=Red,  marker='D', linestyle='None')
    ax.plot( np.nan, np.nan, color=Blue, marker='D', linestyle='None')
    ax.plot( np.nan, np.nan, color=Red,  marker='o', linestyle='None')
    ax.plot( np.nan, np.nan, color=Blue, marker='o', linestyle='None')
    ax.legend( [ 'Theo.-Stab.', 'Theo.-Vol.', 'MDD.-Stab.', 'MDD.-Vol.', 
                 'GAD.-Stab.', 'GAD.-Stab.', 'CON.-Stab.', 'CON.-Vol.'])
    ax.set_axis_off()
    
    ax = axes[ 1, 0]
    for cond, cl in zip(conds,colors):
        ax.plot( np.mean(rdcurves[cond][0],0), np.mean(rdcurves[cond][1],0), color=cl, linewidth=3)
    ax.scatter( pi_comp_dict['CON-Stab'], EQ_dict['CON-Stab'], color=Red,  marker='o')
    ax.scatter( pi_comp_dict[ 'CON-Vol'], EQ_dict[ 'CON-Vol'], color=Blue, marker='o')
    ax.set_title( f'Control Stab-Vol ({title_str})')
    ax.set_ylim( [ .2, .6])
    
    ax = axes[ 1, 1]
    for cond, cl in zip(conds,colors):
        ax.plot( np.mean(rdcurves[cond][0],0), np.mean(rdcurves[cond][1],0), color=cl, linewidth=3)
    ax.scatter( pi_comp_dict['MDD-Stab'], EQ_dict['MDD-Stab'], color=Red,  marker='x')
    ax.scatter( pi_comp_dict[ 'MDD-Vol'], EQ_dict[ 'MDD-Vol'], color=Blue, marker='x')
    ax.scatter( pi_comp_dict['GAD-Stab'], EQ_dict['GAD-Stab'], color=Red,  marker='D')
    ax.scatter( pi_comp_dict[ 'GAD-Vol'], EQ_dict[ 'GAD-Vol'], color=Blue, marker='D')
    ax.set_title( f'Patient Stab-Vol ({title_str})')
    ax.set_yticks([])
    ax.set_ylim( [ .2, .6])

    try:
        plt.savefig( f'{path}/figures/RDfigure.png', dpi=dpi)
    except:
        os.mkdir( f'{path}/figures')
        plt.savefig( f'{path}/figures/RDfigure.png', dpi=dpi)

def vis_model_cmp( data_set):
    nr = 2
    nc = 1 
    fig, axes = plt.subplots( nr, nc, figsize=(5*nr, 5*nc))
    plt.rcParams.update({'font.size': 12})

    model_lst = [ 'model1', 'model2', 'model7', 
                  'model8', 'model11', 
                  'modelE1', 'modelE2', 'RRmodel']

    ax = axes[ 0]
    c1_tab = pd.read_csv( f'{path}/tables/criter1-{data_set}.csv')
    ax.bar( model_lst, c1_tab.iloc[0, 1:].values)
    ax.set_title('Preferred model by N subjects')

    ax = axes[ 1]
    c2_tab = pd.read_csv( f'{path}/tables/criter2-{data_set}.csv')
    ax.bar( model_lst, c2_tab.iloc[0, 1:].values)
    #ax.set_title('NLL')
    ax.set_ylim( [ 14000, 18000])
    
    plt.savefig( f'{path}/figures/model_cmp.png', dpi=dpi)

def ttest_table():

    ## Get variable
    roi_vars = [ 'MDD-Stab', 'MDD-Vol', 
                 'GAD-Stab', 'GAD-Vol',
                 'CON-Stab', 'CON-Vol',
                     'Stab',     'Vol',
                      'MDD',     'GAD',    'CON']
    
    ## Get policy under each conditions
    




 
if __name__ == '__main__':

    ## STEP0: CHOOSE DATA SET AND MODEL 
    data_set = 'rew_data_exp1'
    model    = 'RRmodel'
  
    ## STEP1: split data, calculate RD curves
    EQ_dict, pi_comp_dict = split_data( data_set, model)
    with open(f'{path}/data/rdcurves-{data_set}.pkl', 'rb') as handle:
        rdcurves = pickle.load( handle)

    ## STEP2: SHOW FIGURES
    vis( rdcurves, (EQ_dict, pi_comp_dict), 'sim') 

    ## STEP3: MODEL COMPARISION
    vis_model_cmp( data_set)

