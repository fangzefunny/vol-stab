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

def Rate_Reward( data, prior=.1, wrong_prior=True):
    '''Analyze the data

    Analyze the data to get the rate distortion curve,

    Input:
        data

    Output:
        Theoretical rate and distortion
        Empirical rate and distortion 
    '''
 
    # prepare an array of tradeoff
    betas = np.logspace( np.log10(.1), np.log10(10), 50)

    # create placeholder
    # the challenge of creating the placeholder
    # is that the length of the variable change 
    # in each iteration. To handle this method, 
    # my strategy is to create a matrix with 
    # the maxlength of each variable and then, use 
    # nanmean to summary the variables
    
    # get the number of subjects
    num_sub = len(data.keys())
    max_setsize = 5
    
    # create a placeholder
    results = dict()
    summary_Rate_data = np.empty( [ num_sub, max_setsize, 2]) + np.nan
    summary_Val_data  = np.empty( [ num_sub, max_setsize, 2]) + np.nan
    summary_Rate_theo = np.empty( [ num_sub, len(betas), max_setsize,]) + np.nan
    summary_Val_theo  = np.empty( [ num_sub, len(betas), max_setsize,]) + np.nan

    # run Blahut-Arimoto
    for subi, sub in enumerate(data.keys()):

        #print(f'Subject:{subi}')
        sub_data  = data[ sub]
        blocks    = np.unique( sub_data.block) # all blocks for a subject
        setsize   = np.zeros( [len(blocks),])
        Rate_data = np.zeros( [len(blocks),])
        Val_data  = np.zeros( [len(blocks),])
        Rate_theo = np.zeros( [len(blocks), len(betas)])
        Val_theo  = np.zeros( [len(blocks), len(betas)])
        #errors    = np.zeros( [len(blocks),])
        #bias_state= np.zeros( [len(blocks), 6]) 

        # estimate the mutual inforamtion for each block 
        for bi, block in enumerate(blocks):
            idx      = ((sub_data.block == block) & 
                        (sub_data.iter < 10))
            states   = sub_data.state[idx].values
            actions  = sub_data.action[idx].values
            cor_acts = sub_data.correct_act[idx].values
            rewards  = sub_data.reward[idx].values
            is_sz    = int(sub_data.is_sz.values[0])
            
            # estimate some critieria 
            #errors[bi]    = np.sum( actions != cor_acts) / len( actions)
            Rate_data[bi] = MI_from_data( states, actions, prior, wrong_prior=wrong_prior)

            Val_data[bi] = np.mean( rewards)

            # estimate the theoretical RD curve
            S_card  = np.unique( states)
            A_card  = range(3)
            nS      = len(S_card)
            nA      = len(A_card)
            
            # calculate distortion fn (utility matrix) 
            Q_value = np.zeros( [ nS, nA])
            for i, s in enumerate( S_card):
                a = int(cor_acts[states==s][0]) # get the correct response
                Q_value[ i, a] = 1

            # init p(s) 
            p_s     = np.zeros( [ nS, 1])
            for i, s in enumerate( S_card):
                p_s[i, 0] = np.mean( states==s)
            p_s += np.finfo(float).eps
            p_s = p_s / np.sum( p_s)
            
            # run the Blahut-Arimoto to get the theoretical solution
            for betai, beta in enumerate(betas):
                
                # get the optimal channel for each tradeoff
                pi_a1s, p_a = Blahut_Arimoto( -Q_value, p_s,
                                              beta, update_prior=(1-wrong_prior))
                # calculate the expected distort (-utility)
                # EU = ∑_s,a p(s)π(a|s)Q(s,a)
                theo_util  = np.sum( p_s * pi_a1s * Q_value)
                # Rate = β*EU - ∑_s p(s) Z(s) 
                # Z(s) = log ∑_a p(a)exp(βQ(s,a))  # nSx1
                # Zstate     = logsumexp( beta * Q_value + np.log(p_a.T), 
                #                     axis=-1, keepdims=True)
                #theo_rate  = beta * theo_util - np.sum( p_s * Zstate)
                theo_rate   = np.sum( p_s * pi_a1s * 
                   ( np.log( pi_a1s + eps_) - np.log( p_a.T + eps_)))

                # record
                Rate_theo[ bi, betai] = theo_rate
                Val_theo[ bi, betai]  = theo_util

            setsize[bi] = len(np.unique( states))

        for zi, sz in enumerate([ 2, 3, 4, 5, 6]):
            summary_Rate_data[ subi, zi, is_sz] = np.nanmean( Rate_data[ (setsize==sz),])
            summary_Val_data[ subi, zi, is_sz]  = np.nanmean(  Val_data[ (setsize==sz),])
            summary_Rate_theo[ subi, :, zi] = np.nanmean( Rate_theo[ (setsize==sz), :],axis=0)
            summary_Val_theo[ subi, :, zi]  = np.nanmean(  Val_theo[ (setsize==sz), :],axis=0)

    # prepare for the output 
    results[ 'Rate_theo'] = np.nanmean( summary_Rate_theo, axis=0)
    results[  'Val_theo'] = np.nanmean(  summary_Val_theo, axis=0)
    results[ 'Rate_data'] = summary_Rate_data
    results[  'Val_data'] = summary_Val_data

    return results