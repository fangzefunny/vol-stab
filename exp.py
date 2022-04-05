import os
import numpy as np 
from scipy.special import softmax 
from utils.viz import *
import matplotlib.pyplot as plt 
import seaborn as sns 

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')


def check_exp2( beta2=6):
    pX = np.linspace(.1, 1, 20)
    pY = np.linspace(.1, 1, 5)
    beta1s = np.linspace(1, 10, 5)
    prob_prob = np.zeros( [ len(pX), len(pY), len(beta1s)])
    exp_prob  = np.zeros( [ len(pX), len(pY), len(beta1s)])
    for b, beta1 in enumerate( beta1s):
        for i, px in enumerate(pX):
            for j, py in enumerate(pY):
                p1 = np.array([[ px, 1-px]])
                p2 = np.array([[py, 1-py],
                            [1-py, py]])
                prob_prob[i,j,b] = (softmax( beta1*p1,axis=1) @ 
                            softmax( beta2*p2,axis=1))[0,0]
                exp_prob[i,j,b]  = softmax( beta1 * p1@ 
                            softmax( beta2*p2,axis=1), axis=1)[0,0]
    fig, axs = plt.subplots( len(pY), len(pY), figsize=(len(pY)*3.5, len(pY)*3.5))
    for j, py in enumerate(pY):
        for b, beta1 in enumerate(beta1s):
            ax = axs[b, j]
            l1 = sns.lineplot( x=pX, y=prob_prob[:,j, b], 
                            lw=3, color=Red, label='soft(py)@soft(px)', ax=ax)
            l2 = sns.lineplot( x=pX, y=exp_prob[:,j, b], 
                            lw=3, color=Blue, label='soft(py@soft(px))', ax=ax)
            l1.legend(fontsize=11)
            l2.legend(fontsize=11)  
            ax.set_title( f'p(y)={py},β={beta1}')  
            ax.set_xlabel( 'px', fontsize=15)
            ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    plt.savefig( f'{path}/figures/check_exp2.png', dpi=250)


def check_exp( beta=12, w=.3, gamma=4):
    ps = np.linspace( .01, 1, 40)
    outcome = np.zeros( [ 4, len( ps)])
    for i, p in enumerate( ps):
        # before normalize 
        outcome[ 0, i] = p 
        # after normalize
        p_soft = softmax( beta*np.array([p, 1-p]) )
        outcome[ 1, i] = p_soft[0]
        # lapse 
        p_lapse = (1-w)*np.array([p, 1-p]) + w*1/2
        outcome[ 2, i] = p_lapse[0]
        # prospect
        p_prosect = np.exp(-(-np.log(p))**gamma)
        outcome[ 3, i] = p_prosect


    plt.figure( figsize=(4.5, 4))
    l1 = sns.lineplot( x=outcome[0,:], y=outcome[0,:], 
                        lw=3, color=Red, label='Raw')
    l2 = sns.lineplot( x=outcome[0,:], y=outcome[1,:], 
                        lw=3, color=Blue, label=f'Logistic, β={beta}')
    l3 = sns.lineplot( x=outcome[0,:], y=outcome[2,:], 
                        lw=3, color=Green, label=f'Lapse, w={w}')
    l4 = sns.lineplot( x=outcome[0,:], y=outcome[3,:], 
                        lw=3, color=Purple, label=f'prospect, γ={gamma}')
    l1.legend(fontsize=11)
    l2.legend(fontsize=11)
    l3.legend(fontsize=11)
    l4.legend(fontsize=11)
    plt.xlabel( 'prob', fontsize=15)
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig( f'{path}/figures/check_exp.png', dpi=250)
    
if __name__ == '__main__':

    check_exp2()
