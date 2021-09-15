import numpy as np 
import pandas as pd 
from scipy.optimize import minimize

eps_ = 1e-18

class subj:
    '''Out loop of the fit

    This class can instantiate a dynmaic 
    decision-making model. 
    '''

    def __init__( self, brain):
        self.brain = brain 

    def assign_data( self, data, act_dim):
        self.train_data = data
        self.state_dim  = len( data[0].state.unique())
        self.act_dim    = act_dim  

    def mle_loss(self, params):
        '''Calculate total NLL of the data 
           (over all samples)
        '''
        tot_nll = 0.
        for i in range(len(self.train_data)):            
            data = self.train_data[i]
            tot_nll += self._sample_like( data, params)        
        return tot_nll

    def _sample_like( self, data, params):
        '''Likelihood for one sample

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
            state = int( data.state[t])
            act   = int( data.action[t])
            rew   = obs[ act]
            # store 
            brain.memory.push( obs, state, act, rew, t)
            # evaluate: log π(a|xt)
            NLL += - np.log( brain.eval_act( obs, act) + eps_)
            # model update 
            brain.update()
        
        return NLL

    def fit( self, data, bnds, seed, init=[]):
        '''Core fn used to do one fit
        '''

        # prepare for the fit
        np.random.seed( seed)
        act_dim = 2
        self.assign_data( data, act_dim)
        n_params = len( bnds)

        ## init the optimization params
        # usually there is no init
        if len( init) == 0:
            # init parameters
            param0 = list() 
            for i in range( n_params):
                i0 = bnds[i][0] + ( bnds[i][1] - bnds[i][0]
                                  ) * np.random.rand()
                param0.append( i0)
        # sometimes we fix initalization
        else: 
            param0 = init 
    
        ## start fit 
        print( f'''
                Init with params:
                    {param0} 
                ''')
        res = minimize( self.mle_loss,          # optimize object    
                        param0,                 # initialization
                        bounds= bnds,           # param bound 
                        options={'disp': False} # verbose loss
                        )
        print( f'''
                Fitted params: {res.x}
                MLE loss: {res.fun}
                ''')
        
        # select the optimal param set 
        param_opt = res.x
        loss_opt  = res.fun
        
        return param_opt, loss_opt

    def pred( self, params, data=False):
        '''Model prediction/simulation

        Default:
            use the train data and generate train target
        '''

        # load the train data if no test data
        if data == False:
            data = self.train_data

        # Create a dataframe to record simulation
        # outcome
        col_name = [ 'sub_id', 'action', 'state','act_acc', 
                     'mag0', 'mag1', 'b_type', 'rew',
                     'p_s','pi_0', 'pi_1', 'nll',
                     'pi_comp', 'psi_comp']

        # Loop over sample to collect data 
        sim_data = pd.DataFrame( columns=col_name)
        for i in data:
            input_sample = data[i].copy()
            sim_sample = self._pred_sample( input_sample, params)
            sim_data = pd.concat( [ sim_data, sim_sample], axis=0, sort=True)
        
        return sim_data 

    def _pred_sample( self, data, params):

        state_dim = len( data.state.unique())
        act_dim = 2
        brain = self.brain( state_dim, act_dim, params) 
        data['act_acc']    = float('nan')
        data['p_s']        = float('nan')
        data['pi_0']       = float('nan')
        data['pi_1']       = float('nan')
        data['nll']        = float('nan') 
        data['rew']        = float('nan')
        data['pi_comp']    = float('nan')
        data['psi_comp']   = float('nan') 

        for t in range( data.shape[0]):

            # obtain st, at, and rt
            mag0        = int( data.mag0[t])
            mag1        = int( data.mag1[t])
            obs         = [ mag0, mag1]
            state       = int( data.state[t])
            human_act   = int( data.action[t])
            correct_act = int( data.mag0[t] <= data.mag1[t])
            act         = brain.get_action( obs)
            rew         = obs[ act]
            
            # evaluate action: get p(a|xt)
            pi_a1x = brain.eval_act( state, act)
            ll     = brain.eval_act( state, human_act)

            # record some vals
            # general output 
            data['act'][t]            = act
            data['rew'][t]            = rew
            data['act_acc'][t]        = pi_a1x
            data['p_s'][t]            = self.p_s
            data['nll'][t]            = - np.log( ll + eps_)
            
            # add some model specific output
            try: 
                data['pi_0'][t]       = brain.pi_0
            except: 
                pass 
            try: 
                data['pi_1'][t]       = brain.pi_1
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

            # store to memory     
            brain.memory.push( obs, state, act, rew, t)
            
            # model update
            brain.update()            
            
        return data








            


