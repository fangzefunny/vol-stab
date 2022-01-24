import os
import pickle 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 

from scipy.stats import ttest_ind

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
# create the folders for this folder
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')
# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors    = [ Blue, Red, Green, Yellow, Purple]
sns.set_style("whitegrid", {'axes.grid' : False})

# image dpi
dpi = 250

def val_fit( outcomes, model):
    data = outcomes[model]
    fig, axs = plt.subplots( 1, 2, figsize=( 6, 2.5))
    d_lst = [data[f'human-model-Stab'][0], data[f'human-model-Vol'][0]]
    ax = axs[0]
    for j, d in enumerate(d_lst):
            ax.scatter( j*np.ones_like(d), d, s=20, color=colors[j], alpha=.4)
            ax.errorbar( [j-.1,j,j+.1], [np.nanmean(d)]*3, 
                        [0,np.nanstd(d)/np.sqrt(len(d)),0], color='k') 
    ax.set_xticks([0, 1,])
    ax.set_xlim([-.5, 1.5])
    ax.set_xticklabels( ['Stable', 'Volatile'], fontsize=16)
    ax.set_ylabel( 'Avg. human reward', fontsize=16)
    print( f'Avg. huamn rew ttest: {ttest_ind( d_lst[0], d_lst[1])}')
    d_lst = [data[f'human-model-Stab'][1], data[f'human-model-Vol'][1]]
    ax = axs[1]
    for j, d in enumerate(d_lst):
            ax.scatter( j*np.ones_like(d), d, s=20, color=colors[j], alpha=.4)
            ax.errorbar( [j-.1,j,j+.1], [np.nanmean(d)]*3, 
                        [0,np.nanstd(d)/np.sqrt(len(d)),0], color='k') 
    ax.set_xticks([0, 1,])
    ax.set_xlim([-.5, 1.5])
    ax.set_xticklabels( ['Stable', 'Volatile'], fontsize=16)
    ax.set_ylabel( 'Avg. model reward', fontsize=16)
    print( f'Avg. {model} rew ttest: {ttest_ind( d_lst[0], d_lst[1])}')
    plt.tight_layout()

def show_rd_curves( outcomes, model, mode):
    '''Show the rate-distortion curve 
    '''
    data = outcomes[model]
    groups = [ 'Stab', 'Vol']
    fig, axs = plt.subplots( 2, 2, figsize=( 6, 5))
    # rate distortion curve
    ax = axs[ 0, 0]
    ax.scatter( data[f'eq-pi_comp-Stab-{mode}'][1], 
                 data[f'eq-pi_comp-Stab-{mode}'][0],
                 color=Blue, s=60)
    ax.scatter( data[f'eq-pi_comp-Vol-reg'][1],
                 data[f'eq-pi_comp-Vol-reg'][0],
                 color=Red, s=60)
    ax.legend( groups)
    ax.set_xlabel('Policy Complexiy', fontsize=16)
    ax.set_ylabel('Expected reward', fontsize=16)
    # human rew
    ax = axs[ 0, 1]
    ax.set_axis_off()
    # policy complexity 
    d_lst = [ data[f'eq-pi_comp-Stab-{mode}'][1], data[f'eq-pi_comp-Vol-reg'][1]]
    ax = axs[ 1, 0]
    for j, d in enumerate(d_lst):
            ax.scatter( j*np.ones_like(d), d, s=20, color=colors[j], alpha=.4)
            ax.errorbar( [j-.1,j,j+.1], [np.nanmean(d)]*3, 
                        [0,np.nanstd(d)/np.sqrt(len(d)),0], color='k') 
    ax.set_xticks([0, 1,])
    ax.set_xlim([-.5, 1.5])
    ax.set_xticklabels( ['Stable', 'Volatile'], fontsize=16)
    ax.set_ylabel( 'Avg. policy complexity', fontsize=16)
    print( f'{mode} policy complexity ttest: {ttest_ind( d_lst[0], d_lst[1])}')
    # expected reward
    d_lst = [ data[f'eq-pi_comp-Stab-{mode}'][0], data[f'eq-pi_comp-Vol-reg'][0]]
    ax = axs[ 1, 1]
    for j, d in enumerate(d_lst):
            ax.scatter( j*np.ones_like(d), d, s=20, color=colors[j], alpha=.4)
            ax.errorbar( [j-.1,j,j+.1], [np.nanmean(d)]*3, 
                        [0,np.nanstd(d)/np.sqrt(len(d)),0], color='k') 
    ax.set_xticks([0, 1,])
    ax.set_xlim([-.5, 1.5])
    ax.set_xticklabels( ['Stable', 'Volatile'], fontsize=16)
    ax.set_ylabel( 'Avg. expected reward', fontsize=16)
    print( f'{mode} expected reward ttest: {ttest_ind( d_lst[0], d_lst[1])}')
    plt.tight_layout()
    
def show_RR_params( outcomes, model):
    '''Show the parameters summary 
    '''
    data = outcomes[ model]
    params = [ 'alpha_s', 'alpha_a', 'beta']
    groups = [ 'Stab', 'Vol']
    params_name = [ r'$\alpha_s$', r'$\alpha_a$', r'$\beta$']
    fig, axs = plt.subplots( 2, 2, figsize=( 6, 5))
    
    al = .3
    for i, param in enumerate(params):
        ax = axs[i//2, i%2]
        d_lst = [ data[f'{param}-block'][0], data[f'{param}-block'][1]]
        colors = [ Blue, Red,]
        for j, d in enumerate(d_lst):
            ax.scatter( j*np.ones_like(d), d, s=20, color=colors[j], alpha=al)
            ax.errorbar( [j-.1,j,j+.1], [np.mean(d)]*3, 
                        [0,np.std(d)/np.sqrt(len(d)),0], color='k') 
        ax.set_xticks([0, 1,])
        ax.set_xlim([-.5, 1.5])
        ax.set_xticklabels( ['Stable', 'Volatile'],fontsize=16)
        ax.set_ylabel( params_name[i],fontsize=16)
        print( f't test for {model} {param}: {ttest_ind( d_lst[0], d_lst[1])}')

        ax = axs[1, 1]
        ax.set_axis_off()

    fig.tight_layout()
   

if __name__ == '__main__':

    ## STEP1: EXPLORE RAW DATA
    
    ## STEP1: MODEL-BASED ANALYSIS
    datasets = ['exp1_rew', 'rew_con']
    models   = ['RDModel2', 'model11']
    for dataset in datasets:
        for model in models:
            fname = f'{path}/analyses/analyses-{dataset}.pkl'
            with open( fname, 'rb')as handle:
                    outcomes = pickle.load( handle)
            val_fit( outcomes, model)
            plt.savefig( f'{path}/figures/fit_validate-{dataset}-model={model}.png', dpi=500) 
            show_RR_params( outcomes, model)
            plt.savefig( f'{path}/figures/param_smary-{dataset}-model={model}.png', dpi=500) 
            if model != 'model11':
                show_rd_curves( outcomes, model, 'reg')
                plt.savefig( f'{path}/figures/RD_curves-{dataset}-model={model}.png', dpi=500) 
