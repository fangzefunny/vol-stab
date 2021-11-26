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

    def pred( self, params, data=False):
        '''Model prediction/simulation

        Default:
            use the train data and generate train target
        '''
        # subject list 
        subj_lst = data.keys()

        # Create a dataframe to record simulation
        # outcome
        col_name = [ 'sub_id', 'group', 'action', 'state','act_acc', 
                     'mag0', 'mag1', 'b_type', 'rew', 'human_act',
                     'p_s','pi_0', 'pi_1', 'nll',
                     'pi_comp', 'psi_comp', 'EQ']

        # Loop over sample to collect data 
        sim_data = pd.DataFrame( columns=col_name)
        for subj in subj_lst:
            datum = data[subj]
            input_sample = datum.copy()
            input_sample['human_act'] = input_sample['action'].copy()
            input_sample['action'] = np.nan 
            sim_sample = self._pred_sample( input_sample, params)
            sim_data = pd.concat( [ sim_data, sim_sample], axis=0, sort=True)
        
        return sim_data 

    def _pred_sample( self, data, params):

        state_dim = len( data.state.unique())
        act_dim = 2
        brain = self.brain( state_dim, act_dim, params) 
        data['act_acc']    = np.nan
        data['p_s']        = np.nan
        data['pi_0']       = np.nan
        data['pi_1']       = np.nan
        data['nll']        = np.nan
        data['rew']        = np.nan
        data['pi_comp']    = np.nan
        data['psi_comp']   = np.nan
        data['EQ']         = np.nan

        for t in range( data.shape[0]):

            # obtain st, at, and rt
            mag0        = data.mag0[t]
            mag1        = data.mag1[t]
            obs         = [ mag0, mag1]
            state       = int( data.state[t])
            human_act   = int( data.human_act[t])
            ctxt        = int( data.b_type[t])
            correct_act = int( data.mag0[t] <= data.mag1[t])

            # plan act
            brain.plan_act( obs)
            act         = brain.get_act()
            rew         = obs[ act] * (state == act)
            
            # evaluate action: get p(a|xt)
            pi_a1x = brain.eval_act( act)
            ll     = brain.eval_act( human_act)

            # record some vals
            # general output 
            data['action'][t]         = act
            data['rew'][t]            = rew
            data['act_acc'][t]        = pi_a1x
            data['p_s'][t]            = brain.p_s[ 0, 0]
            data['nll'][t]            = - np.log( ll + eps_)
            data['human_act'][t]      = human_act
            
            # add some model specific output
            try: 
                data['pi_0'][t]       = brain.pi[ 0, 0]
            except: 
                pass 
            try: 
                data['pi_1'][t]       = brain.pi[ 1, 0]
            except:
                pass 
            try:
                data['pi_comp'][t]    = brain.pi_comp()
            except:
                pass 
            try:
                data['psi_comp'][t]   = brain.psi_comp()
            except:
                pass
            try: 
                data['EQ'][t]         = brain.EQ( obs)
            except: 
                pass 

            # what to remember 
            mem = { 'ctxt': ctxt, 'obs': obs, 'state': state,
                  'action': act,  'rew': rew,     't': t }    
            brain.memory.push( mem)
            
            # model update
            brain.update()            
            
        return data








            


