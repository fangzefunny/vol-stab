import numpy as np 
import pandas as pd 
import warnings
# ignore this warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from scipy.optimize import minimize

eps_ = 1e-18
max_ = 1e+8

class subj:
    '''Out loop of the fit

    This class can instantiate a dynmaic 
    decision-making model. 
    '''

    def __init__( self, brain, param_priors=None):
        self.brain = brain 
        self.param_priors = param_priors

    def assign_data( self, data, act_dim):
        self.train_data = data
        self.state_dim  = 2#len( data[0].state.unique())
        self.act_dim    = act_dim  

    def loss_fn(self, params, data):
        '''Total likelihood
        log p(D|θ) = -log ∏_i p(D_i|θ)
                   = ∑_i -log p( D_i|θ )
        or Maximum a posterior 
        log p(θ|D) = ∑_i -log p( D_i|θ ) + -log p(θ)
        '''
        tot_loss = [ self._like( params, data[key] \
                   + self._prior( params)) 
                           for key in data.keys()]        
        return np.sum( tot_loss)

    def _like( self, params, data):
        '''Likelihood for one sample

        -log p( D_i|θ )

        In RL, each sample is a block of experiment,
        Because it is independent across experiment.
        '''

        # init 
        NLL = 0.
        brain = self.brain( self.state_dim, self.act_dim, params)

        # loop to estimate the neg log likelihood
        for t in range( data.shape[0]):
            # obtain st and at, 
            mag0  = data.mag0[t]
            mag1  = data.mag1[t]
            obs   = [ mag0, mag1]
            # planning the action 
            brain.plan_act( obs)
            ctxt  = int( data.b_type[t])
            state = int( data.state[t])
            act   = int( data.action[t])
            rew   = obs[ act]
            # store 
            mem = { 'ctxt': ctxt, 'obs': obs, 'state': state,
                  'action': act,  'rew': rew,     't': t }    
            brain.memory.push( mem)
            # evaluate: log π(a|xt)
            NLL += - np.log( brain.eval_act( act) + eps_)
            # leanring stage 
            brain.update()
        
        return NLL

    def _prior( self, params):
        '''Add the prior of the parameters
        '''
        tot_pr = 0.
        if self.param_priors:
            for prior, param in zip(self.param_priors, params):
                tot_pr += -np.max([prior.logpdf( param), -max_])
        return tot_pr

    def fit( self, data, pbnds, bnds=None, seed=2021, 
                init=None, verbose=False,):
        '''Fit the parameter using optimization 
        '''
        # Init params
        if init:
            # if there are assigned params
            param0 = init
        else:
            # random init from the possible bounds 
            rng = np.random.RandomState( seed)
            param0 = [pbnd[0] + (pbnd[ 1] - pbnd[0]
                     ) * rng.rand() for pbnd in pbnds]
                     
        ## Fit the params 
        verbose and print( 'init with params: ', param0) 
        res = minimize( self.loss_fn, param0, args=( data), 
                        bounds=bnds, options={'disp': verbose})
        verbose and print( f'''  Fitted params: {res.x}, 
                    MLE loss: {res.fun}''')
        
        return res.x, res.fun 

    def predict( self, params, data=False):
        '''Calculate the predicted trajectories
        using fixed parameters
        '''
        # each sample contains human respose within a block 
        out_data = []
        for i in data.keys():
            input_sample = data[i].copy()
            input_sample['human_act'] = input_sample['action'].copy()
            input_sample = input_sample.drop( columns=['action', 'reward'])
            out_data.append( self.simulate( input_sample, params))
        return pd.concat( out_data, ignore_index=True)

    def simulate( self, data, params):

        ## Init the agent 
        state_dim = len( data.state.unique())
        action_dim = 3
        agent= self.agent( state_dim, action_dim, self.rng, params) 
        
        ## init a blank dataframe to store simulation
        col = [ 'action', 'reward', 'prob', 
                'pi_comp', 'psi_comp', 
                'tradeoff', 'cog_load', 'cog_vio',
                'pi_weights', 'psi_weights']
        init_mat = np.zeros([ data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame( init_mat, columns=col)  

        for t in range( data.shape[0]):

            # obtain st, at, and rt
            state = int(data['state'][t])
            correct_act = int(data['correctAct'][t])
            agent.plan_act( state)
            action = agent.get_act()
            reward = np.sum( action == correct_act)
            
            # evaluate action: get p(ai|S = si)
            pi_a1x = agent.eval_act( correct_act)

            # record some vals
            pred_data['action'][t] = action
            pred_data['reward'][t] = reward
            pred_data['prob'][t]   = pi_a1x

            # record some important variable
            if agent.pi_comp() is not None: pred_data['pi_comp'][t]  = agent.pi_comp()
            if agent.psi_comp() is not None: pred_data['psi_comp'][t] = agent.psi_comp()
            if agent.get_tradeoff() is not None: pred_data['tradeoff'][t] = agent.get_tradeoff()
            if agent.get_cogload() is not None: pred_data['cog_load'][t] =  agent.get_cogload()
            if agent.get_cogvio() is not None: pred_data['cog_vio'][t] =  agent.get_cogvio()

            # record the policy
            p_pi  = ''
            for s in range(agent.pi.shape[0]): 
                p = ','.join([str(np.round( i,4)) for i in agent.pi[ s, :]])
                p_pi += '\\' * (s > 0) + p
            pred_data['pi_weights'] = p_pi 

            # the perception (if any)
            if agent.psi is not None:
                p_psi = ','.join([str(np.round(i,4)) for i in agent.psi[ state, :]])
                pred_data['psi_weights'] = p_psi

            # store 
            mem = { 'stim': state, 'act': action, 'rew': reward, 't': t+1 } 
            agent.memory.push( mem)   
            
            # model update
            agent.update()     

        ## Merge the data into a large 
        pred_data = pred_data.dropna( axis=1, how='all')
        data = pd.concat( [ data, pred_data], axis=1)       
            
        return data








            


