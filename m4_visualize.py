import os
import pickle 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import statsmodels.stats.weightstats as st

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

def ttest_table( data_dict):

    ## Get variable
    roi_vars = data_dict.keys()
    alpha1   = 0.05 
    alpha2   = 0.01
    
    ## Get policy under each conditions
    h_map = np.zeros( [len(roi_vars), len(roi_vars)])
    for i, v1 in enumerate(roi_vars):
        d0 = data_dict[v1]
        for j, v2 in enumerate(roi_vars):
            d1 = data_dict[v2]
            t, p_two, _ = st.ttest_ind( d0, d1, usevar='unequal')
            h_map[ i, j] =  1 * (p_two <= alpha1/2) * np.sign(t) +\
                            1 * (p_two <= alpha2/2) * np.sign(t)
    
    # remove the upper half
    return h_map 

def EU_vs_symp( data_set):
    '''
    E_p(s,a)[U(s,a)] = 1/N ∑_i U(s_i,a_i)

    U(s,a) 
    '''
    ## Load data
    with open(f'{path}/data/{data_set}.pkl', 'rb') as handle:
        data = pickle.load( handle)

    ## Define vars and placeholders,
    # each conditions containst two cases:
    # case 0: general
    # case 1: stable
    # case 2: volatile 
    subj_lst = data.keys()
    out_data = {'Gen':  { 'CON': [], 'MDD': [], 'GAD': []}, 
                'Stab': { 'CON': [], 'MDD': [], 'GAD': []}, 
                'Vol':  { 'CON': [], 'MDD': [], 'GAD': []}} 

    ## Loop to group the data 
    for subj in subj_lst:
        sub_data = data[subj] 
        gp  = sub_data['group'][0]
        cor = (sub_data['action'] == sub_data['state'])
        mag = (1 - sub_data['state'].values) * sub_data['mag0'].values \
                 + sub_data['state'].values * sub_data['mag1']
        rew = cor * mag

        out_data['Gen' ][gp].append( np.mean( rew))
        out_data['Stab'][gp].append( np.mean( rew[ sub_data['b_type']==1]))
        out_data['Vol' ][gp].append( np.mean( rew[ sub_data['b_type']==0]))

    ## Generat figure 1: [ 'Gen', 'Stab', 'Vol']
    nr = 2
    nc = 2 
    fig, axes = plt.subplots( nr, nc, figsize=(4*nr, 4*nr))
    plt.rcParams.update({'font.size': 12})
    g_names = [ 'Gen', 'Stab', 'Vol']
    g_ind   = [ 0, 2, 3]
    for idx, g_key in zip(g_ind, g_names):
        ax = axes[ idx // nc, idx % nc ]
        ax.boxplot( [out_data[g_key][gp] for gp in out_data[g_key].keys()])
        ax.set_xticklabels(list(out_data[g_key].keys()))
        ax.set_ylim( [ .2, .45])
        ax.set_title( g_key)
    axes[ 0, 1].axis('off')
    plt.savefig( f'{path}/figures/raw_data1.png', dpi=200)

    ## Generate figure 1': [ 'CON', 'MDD', 'GAD']
    g_names = [ 'CON', 'MDD', 'GAD']
    xs      = [ 'Gen', 'Stab', 'Vol']
    out_dict2 = dict()
    for g1 in g_names:
        out_dict2[g1] = dict()
        for x in xs:
            out_dict2[g1][x] =  out_data[x][g1]
    
    fig, axes = plt.subplots( nr, nc, figsize=(4*nr, 4*nr))
    plt.rcParams.update({'font.size': 12})
    g_ind   = [ 0, 2, 3]
    for idx, g_key in zip(g_ind, g_names):
        ax = axes[ idx // nc, idx % nc ]
        ax.boxplot( [out_dict2[g_key][gp] for gp in out_dict2[g_key].keys()])
        ax.set_xticklabels(list(out_dict2[g_key].keys()))
        ax.set_ylim( [ .2, .45])
        ax.set_title( g_key)
    axes[ 0, 1].axis('off')
    plt.savefig( f'{path}/figures/raw_data2.png', dpi=200)

    ## Heat map for significant
    out_dict3 = dict()
    roi_vars = [ 'MDD-Stab', 'GAD-Stab', 'CON-Stab']
    for var in roi_vars:
        k1, k2 = var.split('-')
        out_dict3[var] = out_dict2[k1][k2]
    fig3 = plt.figure()
    h_map = ttest_table( out_dict3)
    plt.imshow( h_map, cmap='Reds')
    plt.colorbar()
    x_roi_vars = [ var+'\'' for var in roi_vars]
    plt.xticks( range(len(roi_vars)), x_roi_vars, fontsize=6.5, rotation=45)
    plt.yticks( range(len(roi_vars)), roi_vars, fontsize=6.5, rotation=45)
    plt.savefig( f'{path}/figures/tt_heatmap1.png', dpi=200)

    roi_vars = [ 'MDD-Vol',  'GAD-Vol',  'CON-Vol']
    out_dict3 = dict()
    for var in roi_vars:
        k1, k2 = var.split('-')
        out_dict3[var] = out_dict2[k1][k2]
    fig3 = plt.figure()
    h_map = ttest_table( out_dict3)
    plt.imshow( h_map, cmap='Reds')
    plt.colorbar()
    x_roi_vars = [ var+'\'' for var in roi_vars]
    plt.xticks( range(len(roi_vars)), x_roi_vars, fontsize=6.5, rotation=45)
    plt.yticks( range(len(roi_vars)), roi_vars, fontsize=6.5, rotation=45)
    plt.savefig( f'{path}/figures/tt_heatmap1_1.png', dpi=200)


    ## Heatmap for general
    roi_vars2 = [    'Stab',     'Vol',     'Gen',]
    out_dict4 = dict()
    for var in roi_vars2:
        lst = []
        for k in out_data[var].keys():
            lst += out_data[var][k]
        out_dict4[var] = lst 
    fig4 = plt.figure()
    h_map = ttest_table( out_dict4)
    plt.imshow( h_map, cmap='Reds')
    plt.colorbar()
    x_roi_vars = [ var+'\'' for var in roi_vars2]
    plt.xticks( range(len(roi_vars2)), x_roi_vars, fontsize=6.5, rotation=45)
    plt.yticks( range(len(roi_vars2)), roi_vars2, fontsize=6.5, rotation=45)
    plt.savefig( f'{path}/figures/tt_heatmap2.png', dpi=200)


    roi_vars3 = [  'MDD',     'GAD',     'CON']
    out_dict5 = dict()
    for var in roi_vars3:
        lst = []
        for k in out_dict2[var].keys():
            lst += out_dict2[var][k]
        out_dict5[var] = lst 
    fig4 = plt.figure()
    h_map = ttest_table( out_dict5)
    plt.imshow( h_map, cmap='Reds')
    plt.colorbar()
    x_roi_vars = [ var+'\'' for var in roi_vars3]
    plt.xticks( range(len(roi_vars3)), x_roi_vars, fontsize=6.5, rotation=45)
    plt.yticks( range(len(roi_vars3)), roi_vars3, fontsize=6.5, rotation=45)
    plt.savefig( f'{path}/figures/tt_heatmap3.png', dpi=200)
    
   


 
if __name__ == '__main__':

    ## STEP0: CHOOSE DATA SET AND MODEL 
    data_set = 'rew_data_exp1'
    model    = 'RRmodel'
  
    ## STEP1: split data, calculate RD curves
    EQ_dict, pi_comp_dict = split_data( data_set, model)
    with open(f'{path}/data/rdcurves-{data_set}.pkl', 'rb') as handle:
        rdcurves = pickle.load( handle)

    ## STEP2: EXPLORE RAW DATA 
    EU_vs_symp( data_set)

    # ## STEP2: SHOW FIGURES
    # vis( rdcurves, (EQ_dict, pi_comp_dict), 'sim') 

    # ## STEP3: MODEL COMPARISION
    # vis_model_cmp( data_set)

