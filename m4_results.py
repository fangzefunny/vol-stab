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

def viz_task():
    plt.figure( figsize=( 6.5, 2.5))
    plt.plot( np.arange(1,91), np.ones([90])*.7,
                '--', color=Blue, linewidth=2)
    plt.fill_between( np.arange(1,91), np.ones([90])*1,
                color=Blue, alpha=.1)
    plt.vlines( x=91, ymin=.2, ymax=.7,
                linestyles='dashed', color=Red, linewidth=2)
    plt.plot( np.arange(91,111), np.ones([20])*.2,
                '--', color=Red, linewidth=2)
    plt.vlines( x=111, ymin=.2, ymax=.8,
                linestyles='dashed', color=Red, linewidth=2)
    plt.plot( np.arange(111,131), np.ones([20])*.8,
                '--', color=Red, linewidth=2)
    plt.vlines( x=131, ymin=.2, ymax=.8,
                linestyles='dashed', color=Red, linewidth=2)
    plt.plot( np.arange(131,151), np.ones([20])*.2,
                '--', color=Red, linewidth=2)
    plt.vlines( x=151, ymin=.2, ymax=.8,
                linestyles='dashed', color=Red, linewidth=2)
    plt.plot( np.arange(151,171), np.ones([20])*.8,
                '--', color=Red, linewidth=2)
    plt.vlines( x=171, ymin=.2, ymax=.8,
                linestyles='dashed', color=Red, linewidth=2)
    plt.plot( np.arange(171,181), np.ones([10])*.2,
                '--', color=Red, linewidth=2)
    plt.text( 30, .9, 'Stable', fontsize=16)
    plt.text( 120, .9, 'Volatile', fontsize=16)
    plt.xlabel( 'Trials', fontsize=16)
    plt.xlim( [ 1, 180])
    plt.ylim( [ 0, 1.05])
    plt.ylabel( 'The left stimulus \nresults in reward', fontsize=14)
    plt.tight_layout()
    plt.savefig( f'{path}/figures/experiment_paradigm.png', dpi=dpi)
    

def viz_fit_goodness( outcomes, model):
    data = outcomes[model]['RC-anlyses']
    crs  = [ 'rew', 'rew_hat']
    subj = [ 'human', 'model']
    fig, axs = plt.subplots( 1, 2, figsize=( 6, 2.5))
    for i, cr in enumerate(crs): 
        ax = axs[i]
        sns.violinplot( x='Trial type', y=cr, data=data,
                     palette=[ Blue, Red], order=['Stable', 'Volatile'], ax=ax)
        ax.set_xticks([0, 1,])
        ax.set_xlim([-.5, 1.5])
        ax.set_xlabel('')
        ax.set_xticklabels( ['Stable', 'Volatile'], fontsize=14)
        ax.set_ylabel( f'Avg. {subj[i]} reward', fontsize=16)
        ax.set_ylim([ .2, .8])
    plt.tight_layout()

def viz_RC_anlyses( outcomes, model):
    '''Show the rate-distortion curve 
    '''
    data = outcomes[model]['RC-anlyses']
    fig, axs = plt.subplots( 2, 2, figsize=( 6, 5))
    # rate distortion curve
    ax = axs[ 0, 0]
    sns.scatterplot(x='pi_comp', y='EQ', data=data, 
                    palette=[ Blue, Red],
                    s=90, hue='Trial type', 
                    legend=True, ax=ax)
    #ax.legend( title='block type', labels=['Stable', 'Volatile'], fontsize=10)
    ax.set_xlabel('Avg. policy complexity', fontsize=16)
    ax.set_ylabel('Avg. expected reward', fontsize=16)
    ax.set_ylim([ .2, .65])
    # human rew
    ax = axs[ 0, 1]
    sns.scatterplot(x='pi_comp', y='rew_hat', data=data, 
                    palette=[ Blue, Red],
                    s=90, hue='Trial type', 
                    legend=False, ax=ax)
    #ax.legend( title='block type', labels=['Stable', 'Volatile'], fontsize=10)
    ax.set_xlabel('Avg. policy complexity', fontsize=16)
    ax.set_ylabel('Avg. actual reward', fontsize=16)
    ax.set_ylim([ .2, .65])
    # policy complexity 
    ax = axs[ 1, 0]
    sns.violinplot( x='Trial type', y='pi_comp', data=data,
                     palette=[ Blue, Red], 
                     order=[ 'Stable', 'Volatile'], 
                     ax=ax)
    ax.set_xticks([0, 1,])
    ax.set_xlim([-.5, 1.5])
    ax.set_xticklabels( ['Stable', 'Volatile'], fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel( 'Avg. policy complexity', fontsize=16)
    #print( f'{mode} policy complexity ttest: {ttest_ind( d_lst[0], d_lst[1])}')
    # expected reward
    ax = axs[ 1, 1]
    sns.violinplot( x='Trial type', y='rew_hat', data=data,
                     palette=[ Blue, Red], 
                     order=[ 'Stable', 'Volatile'], 
                     ax=ax)
    ax.set_xticks([0, 1,])
    ax.set_xlim([-.5, 1.5])
    ax.set_xticklabels( ['Stable', 'Volatile'], fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel( 'Avg. actual reward', fontsize=16)
    #print( f'{mode} expected reward ttest: {ttest_ind( d_lst[0], d_lst[1])}')
    plt.tight_layout()
    
def viz_params( outcomes, model):
    '''Show the parameters summary 
    '''
    data = outcomes[ model]['params']
    params = [ 'alpha_s', 'alpha_a', 'beta']
    params_name = [ r'$\alpha_s$', r'$\alpha_a$', r'$\beta$']
    fig, axs = plt.subplots( 2, 2, figsize=( 6, 5))
    for i, param in enumerate(params):
        ax = axs[i//2, i%2]
        sns.violinplot( x='Trial type', y=param, data=data,
                     palette=[ Blue, Red], 
                     order=[ 'Stable', 'Volatile'], 
                     ax=ax)
        ax.set_xticks([0, 1,])
        ax.set_xlim([-.5, 1.5])
        ax.set_xticklabels( ['Stable', 'Volatile'],fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel( params_name[i],fontsize=16)
        ax = axs[1, 1]
        ax.set_axis_off()
    fig.tight_layout()

def ttest( data, col, name):
    d1 = data[col][data['Trial type']=='Stable']
    d2 = data[col][data['Trial type']=='Volatile']
    res = ttest_ind( d1, d2)
    print( f'T test for {name}: t={res[0]:.3f}, p={res[1]:.3f}')

def t_tests( outcomes, model):

    data = outcomes[model]
    ## Test for figure 1 
    cols  = ['rew', 'rew_hat']
    names = [ 'human reward', 'model reward']
    for col, name in zip( cols, names):
        ttest( data['RC-anlyses'], col, name)
    ## Test for figure 2
    cols  = ['alpha_s', 'alpha_a', 'beta']
    names = cols
    for col, name in zip( cols, names):
        ttest( data['params'], col, name)
    ## Test for figure 3
    if model != 'model11':
        cols  = [ 'pi_comp', 'rew_hat']
        names = [ 'policy complexity', 'actual reward']
        for col, name in zip( cols, names):
            ttest( data['RC-anlyses'], col, name)

if __name__ == '__main__':

    ## Show experiment paradigm
    viz_task()

    # ## Analyze the data 
    # datasets = ['exp1_rew']
    # models   = ['RDModel2', 'model11']
    # for dataset in datasets:
    #     for model in models:
    #         fname = f'{path}/analyses/analyses-{dataset}.pkl'
    #         with open( fname, 'rb')as handle:
    #                 outcomes = pickle.load( handle)
    #         viz_fit_goodness( outcomes, model)
    #         plt.savefig( f'{path}/figures/fit_validate-{dataset}-model={model}.png', dpi=500) 
    #         viz_params( outcomes, model)
    #         plt.savefig( f'{path}/figures/param_smary-{dataset}-model={model}.png', dpi=500) 
    #         if model != 'model11':
    #             viz_RC_anlyses( outcomes, model)
    #             plt.savefig( f'{path}/figures/RD_curves-{dataset}-model={model}.png', dpi=500) 
    #         t_tests( outcomes, model)