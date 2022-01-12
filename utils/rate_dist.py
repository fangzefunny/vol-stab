#===================================
#       Rate-distortion package
#===================================

## import package
import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
from scipy.optimize import minimize 

## define a small value 
eps_ = 1e-12

## Rate-distortion function
def I( p_x, p_y1x, p_y):
    '''Mutual information
    '''
    Ixy = np.sum( p_x*p_y1x * ( np.log( p_y1x + eps_) 
                              - np.log( p_y.T + eps_)))
    return Ixy

def RD( u_fn, p_x, k, **kwargs):
    '''
    '''
    
    ## Get the hyper-setting of the algorithm
    hyper = { 'n_init': 3,
              'bnd':  [(1/40, 100)],
              'pbnd': (1/10, .6),
              'maxiter': 500,
              'rng': np.random.RandomState( 42)}
    for key in kwargs.keys():
        hyper[key] = kwargs[key]

    ## random init from the bounds
    pbnd = hyper['pbnd']
    opt_val = np.inf
    for _ in range( hyper['n_init']):
        tau0 = pbnd[0] + (pbnd[ 1] - pbnd[0]) * hyper['rng'].rand() 

        ## Search the best params
        res = minimize( pRes, tau0, 
                            args=( u_fn, k, p_x), 
                            bounds=hyper['bnd'],
                            options={'disp': False,
                                    'maxiter': hyper['maxiter']})
        
        if res.fun < opt_val:
            opt_val = res.fun
            opt_x   = res.x
    
    ## Get some values
    tau = opt_x 
    p_x  = p_x.reshape( [ -1, 1])
    p_y1x, p_y = Blahut_Arimoto( u_fn, p_x, tau)
    I_xy = I( p_x, p_y1x, p_y)
    vio = np.max([ 0, I_xy - k]) 

    return p_y1x, p_y, tau, vio

def adaRD( u_fn, p_x, k, **kwargs):
    
    ## Get the hyper-setting of the algorithm
    hyper = { 'n_init': 3,
              'bnd':  [ (1/40, 100),],
              'pbnd': [ (1/20, .6),],
              'rng': np.random.RandomState( 42),
              'maxiter': 500,
              'beta': .8, 
              'tol': 1e-2,
              'lr': .1,
              'p_y1x': None, 
              'p_z1y': None} 
    
    for key in kwargs.keys():
        hyper[key] = kwargs[key]
    
    ## Init for the iteration
    p_x  = p_x.reshape( [ -1, 1])
    r, delta, i = 0, 1e-7, 0
    tau = 1 
    done = False 

    ## init the value 
    nX = u_fn.shape[0]
    p_y1x = hyper['p_y1x'] if hyper['p_y1x'] else np.eye(nX)
         
    ## random init from the bounds
    ## Iterate to get the optimal channel 
    while not done:
        # cache the current τ for termiantion check
        prev_tau = tau 
        # find the best channel given τ
        p_x  = p_x.reshape( [ -1, 1])
        p_y1x, p_y = Blahut_Arimoto( u_fn, p_x, tau)                                 
        # check if statisfaction of the primal constraint
        I_xy = (p_x*p_y1x * ( np.log( p_y1x) - np.log( p_y.T))).sum()
        # Adagrad algorithm: 2-order momentum
        grad = ( I_xy - k)
        r = hyper['beta'] * r + ( 1 - hyper['beta']) * grad ** 2
        lr = hyper['lr'] / ( delta+np.sqrt(r))
        # update τ
        tau = np.clip( tau + lr*grad, 1/1000, 1/eps_)
        # check termination condition
        pRes = norm( grad)
        dRes = norm( tau - prev_tau)
        if np.min( [ pRes, dRes]) < hyper['tol']: done = True
        if i >= hyper['maxiter']:
            print( 'The outer loop excede the maxium iteration')
            done = True 
        # count the iteration 
        i += 1 

    return p_y1x, p_y, tau, (np.max( [ 0, grad]))**2

def pRes( tau, u_fn, k, p_x):
    p_x  = p_x.reshape( [ -1, 1])
    p_y1x, p_y = Blahut_Arimoto( u_fn, p_x, tau)
    I_xy = np.sum( p_x*p_y1x * ( np.log( p_y1x+eps_) 
                                   - np.log( p_y.T+eps_)))
    return ( I_xy - k)**2

def Blahut_Arimoto( u_fn, p_x, tau, **kwargs):
    '''Blahut Arimoto algorithm
    
        Innter loop of channel_given_capcity algorithm.
    '''
     ## Get the hyper-setting of the algorithm
    hyper = { 'tol': 1e-3, 
              'max_iter': 80}
    for key in kwargs.keys():
        hyper[key] = kwargs[key]
    
    ## Init for the iteration
    nx, ny = u_fn.shape[0], u_fn.shape[1]
    p_x  = p_x.reshape( [nx, 1])
    p_y1x = np.ones( [ nx, ny]) / ny 
    p_y  = p_y1x.T @ p_x 
    done, i = False, 0

    ## Iterate to get the optimal channel 
    while not done:
        # cache the current channel for termiantion check
        prev_p_y1x = p_y1x
        #  p(y|x) ∝ exp( U(x,y)/τ + log p(y))
        f_y1x = u_fn/tau + np.log( p_y.T+eps_)
        p_y1x = np.exp( f_y1x - 
                logsumexp( f_y1x, keepdims=True, axis=1)) + eps_
        p_y1x /= p_y1x.sum( 1, keepdims=True)
        # p(y) = ∑_x p(x)p(y|x)
        p_y  = p_y1x.T @ p_x
        # check convergence
        if ( np.sum((p_y1x - prev_p_y1x)**2)) < hyper['tol']:
            done = True
        if i >= hyper['max_iter']:
            print( 'The inter loop excede the maxium iteration')
            done = True 
        # iteration counter
        i += 1

    return p_y1x, p_y
