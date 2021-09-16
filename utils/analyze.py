import numpy as np 
from scipy.special import psi, logsumexp

def nats_to_bits( nats):
    '''Map nats to bits
    '''
    return nats / np.log(2)

def blahut_arimoto( distort, p_x, 
                    beta,
                    tol=1e-3, max_iter=200):
    '''Blahut Arimoto algorithm
    '''
    # init variable for iteration
    nX, nY = distort.shape[0], distort.shape[1]
    p_y1x = np.ones( [ nX, nY]) / nY 
    p_y = ( p_x.T @ p_y1x).T 
    done = False
    i = 0

    while not done:

        # cache the current channel for convergence check
        old_p_y1x = p_y1x 
        
        # p(y|x) ∝ p(y)exp(-βD(x,y)) nXxnY
        log_p_y1x = - beta * distort + np.log( p_y.T)
        p_y1x = np.exp( log_p_y1x - logsumexp( log_p_y1x, axis=-1, keepdims=True))

        # p(y) = ∑_x p(x)p(y|x) nYx1
        p_y = ( p_x.T @ p_y1x).T + np.finfo(float).eps
        p_y = p_y / np.sum( p_y)

        # iteration counter
        i += 1

        # check convergence
        if np.sum(abs( p_y1x - old_p_y1x)) < tol:
            done = True 
        if i >= max_iter:
            #print( f'reach maximum iteration {max_iter}, results might not inaccurate')
            done = True 
    
    return p_y1x, p_y 