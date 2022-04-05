import numpy as np 
import pandas as pd 
import warnings
# ignore this warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from scipy.optimize import minimize

eps_ = 1e-12
max_ = 1e+8

class subj:
    '''Out loop of the fit

    This class can instantiate a dynmaic 
    decision-making model. 
    '''

    def __init__( self, agent, param_priors=None, seed=1234):
        self.agent = agent 
        self.param_priors = param_priors
        self.rng   = np.random.RandomState(seed)

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
        tot_loss = [ self._like( params, data[key])
                   + self._prior( params) for key in data.keys()]        
        return np.sum( tot_loss)

    def _like( self, params, data):
        '''Likelihood for one sample

        -log p( D_i|θ )

        In RL, each sample is a block of experiment,
        Because it is independent across experiment.
        '''

        # init 
        NLL = 0.
        agent = self.agent( 2, 2, self.rng, params)

        # loop to estimate the neg log likelihood
        for t in range( data.shape[0]):
            # obtain st and at, 
            mag0  = data.mag0[t]
            mag1  = data.mag1[t]
            obs   = [ mag0, mag1]
            ctxt  = int( data.b_type[t])
            state = int( data.state[t])
            # planning the action
            mem = { 'ctxt': ctxt, 'obs': obs, 'state': state }    
            agent.memory.push( mem) 
            agent.plan_act()
            act   = int( data.action[t])
            rew   = obs[ act]
            # store 
            mem = { 'act': act,  'rew': rew,     't': t }    
            agent.memory.push( mem)
            # evaluate: log π(a|xt)
            NLL += - np.log( agent.eval_act( act) + eps_)
            # leanring stage 
            agent.update()
        
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
                verbose=False, init=None,):
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

    def predict( self, data, params, ):
        '''Calculate the predicted trajectories
        using fixed parameters
        '''
        # each sample contains human respose within a block 
        out_data = []
        for i in data.keys():
            input_sample = data[i].copy()
            out_data.append( self.simulate( input_sample, params))
        return pd.concat( out_data, ignore_index=True)

    def simulate( self, data, params):

        ## Init the agent 
        state_dim = len( data.state.unique())
        action_dim = 2
        agent= self.agent( state_dim, action_dim, self.rng, params) 
        
        ## init a blank dataframe to store simulation
        col = [ 'rew', 'rew_hat', 'p_s',
                'pi_0', 'pi_1', 'q_a',
                'nll', 'pi_comp', 'EQ']
        init_mat = np.zeros([ data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame( init_mat, columns=col)  

        for t in range( data.shape[0]):
            
             # obtain st and at, 
            mag0      = data['mag0'][t]
            mag1      = data['mag1'][t]
            obs       = [ mag0, mag1]
            state     = int(data['state'][t])
            ctxt      = int(data['b_type'][t])
            # planning the action
            mem = { 'ctxt': ctxt, 'obs': obs, 'state': state }    
            agent.memory.push( mem) 
            agent.plan_act()
            act   = int( data.action[t])
            rew   = obs[ act]
            rew_hat = rew*agent.P_a[ act] if rew>0 else obs[1-act]*agent.P_a[1-act]
            
            # evaluate: log π(a|xt)
            nll = - np.log( agent.eval_act( act) + eps_)

            # record some vals
            pred_data['rew'][t]       = rew
            pred_data['rew_hat'][t]   = rew_hat
            pred_data['p_s'][t]       = agent.p_s[ 0, 0]
            pred_data['nll'][t]       = nll

            # record some important variable
            if agent.pi is not None: 
                pred_data['pi_0'][t] = agent.pi[ 0, 0]
                pred_data['pi_1'][t] = agent.pi[ 1, 0]
            if agent.q_a is not None: pred_data['q_a'][t] = agent.q_a[ 0, 0]
            if agent.pi_comp() is not None: pred_data['pi_comp'][t] = agent.pi_comp()
            if agent.EQ() is not None: pred_data['EQ'][t] = agent.EQ()

            # store 
            mem = { 'act': act,  'rew': rew,     't': t }   
            agent.memory.push( mem)   
            
            # model update
            agent.update()     

        ## Merge the data into a large 
        pred_data = pred_data.dropna( axis=1, how='all')
        data = pd.concat( [ data, pred_data], axis=1)   

        # # show pi_comp
        # pi_comp1 = data['pi_comp'][ data['b_type']==1].mean()
        # pi_comp2 = data['pi_comp'][ data['b_type']==0].mean()
        # sb_id = data['sub_id'][0]
        # print( f'Sub id: {sb_id}')
        # print( f'Stab beta: {agent.beta_stab}; Vol beta: {agent.beta_vol}')
        # print( f'Stab Pi Comp: {pi_comp1}; Vol Pi Comp: {pi_comp2}')    
        return data








            


