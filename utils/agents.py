from ast import If
import numpy as np
from numpy.core.fromnumeric import argmax
from pandas import isna 
from scipy.special import softmax, logsumexp
from scipy.stats import norm, bernoulli, beta 
from utils.rate_dist import RD, I

# get the machine epsilon
eps_ = 1e-12
max_ = 1e+10

def MI( p_s, pi, q_a):
    return np.sum( p_s * pi * ( np.log( pi + eps_) - np.log( q_a.T + eps_)))

#==========================
#       Memory Buffer
#==========================

class simpleBuffer:
    '''Simple Buffer 2.0

    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__( self):
        self.m = {}
        
    def push( self, m_dict):
        for key in m_dict.keys():
            self.m[ key] = m_dict[ key]
        
    def sample( self, *args):
        lst = []
        for key in args:
            lst.append( self.m[ key])
        return lst

#=====================
#     Base Agent
#=====================

class Basebrain:
    
    def __init__( self, nS, nA, rng):
        self.nS  = nS
        self.nA  = nA
        self.rng = rng 
        self._init_beliefs()
        self._init_memory()

    def _load_free_params( self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 

    def _init_beliefs( self):
        self.p_s   = np.ones( [ self.nS, 1]) / self.nS 
        self.v_s   = np.ones( [ self.nS, 1]) / self.nS
        self.q_a   = np.ones( [ self.nA, 1]) / self.nA 
        self.v_a   = np.ones( [ self.nA, 1]) / self.nA 
        self.P_a   = np.ones( [ self.nA,]) / self.nA
        self.pi    = None 

    def _init_memory( self):
        self.memory = simpleBuffer()
    
    def O(self, state):
        I_s = np.zeros( [ self.nS, 1])
        I_s[ state, 0] = 1.
        return I_s

    def Q(self, action):
        I_a = np.zeros( [ self.nA, 1])
        I_a[ action, 0] = 1. 
        return I_a
    
    def plan_act( self):
        '''Generate action given observation
            p(a|xt)
        '''
        return NotImplementedError
        
    def get_act( self):
        '''Sample from p(a|xt)
        '''
        return self.rng.choice( 
            range( self.nA), p=self.P_a)
        
    def eval_act( self, act):
        '''get from p(at|xt)
        '''
        return self.P_a[ act]
        
    def update( self):
        ## Retrieve memory
        self.update_Ps()
    
    def pi_comp( self):
        return None 

    def EQ( self):
        return None 

    def get_U( self):
        # retrieve memory
        obs = self.memory.sample( 'obs')[0]
        # construct utility function: U(s,a)
        mag0, mag1 = obs
        u_sa = np.zeros( [ self.nS, self.nA])
        u_sa[ 0, 0] = mag0
        u_sa[ 1, 1] = mag1
        return u_sa

    def update_Ps( self):
        # retrieve memory
        ctxt, state = self.memory.sample( 'ctxt', 'state')
        # choose parameter set 
        alpha_s = self.alpha_s_stab if ctxt else self.alpha_s_vol
        ## Update p_s
        # δs = It(s) - p_s
        rpe_s = self.O(state) - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += alpha_s * rpe_s 
    
    def update_Pa( self):
        # retrieve memory
        ctxt, act = self.memory.sample( 'ctxt', 'act')
        # choose parameter set 
        alpha_a = self.alpha_a_stab if ctxt else self.alpha_a_vol
        ## Update p_a
        # δs = It(a) - p_a
        rpe_a = self.Q(act) - self.q_a 
        # p_a = p_a + α_a * δa
        self.q_a += alpha_a * rpe_a 

#=====================
#     Gagne Agent
#=====================

class model1( Basebrain):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.gamma        = params[2] # risk preference
        self.beta         = params[3] # inverse temperature

    def plan_act( self):
        # retrieve memory
        obs = self.memory.sample( 'obs')[0]
        # unpack observation
        mag0, mag1 = obs
        # risk preference
        pt = self.p_s[ 0, 0] 
        pt = np.min( [ np.max( [ abs( pt - .5) ** self.gamma * np.sign( pt - .5) 
                       + .5, 0 ] ), 1])
        # diferenece in expected value 
        vt = pt * mag0 - ( 1 - pt) * mag1
        # softmax action selection
        pit = 1 / ( 1 + np.exp( -self.beta * vt))
        # choice probability
        self.P_a = np.array( [ pit, 1 - pit])

class model2( model1):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.lam          = params[2] # mixture of prob and mag
        self.beta         = params[3] # inverse temperature

    def plan_act(self):
        # retrieve memory
        obs = self.memory.sample( 'obs')[0]
        # unpack observation
        mag0, mag1 = obs
        pt = self.p_s[ 0, 0] 
        # mixture of probability and magnitude 
        vt = self.lam * ( pt - ( 1 - pt)) + \
             ( 1 - self.lam) * ( mag0 - mag1)
        # softmax action selection
        pit = 1 / ( 1 + np.exp( -self.beta * vt))
        # choice probability
        self.P_a = np.array( [ pit, 1 - pit])

class model7( model2):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.lam          = params[2] # mixture of prob and mag
        self.r            = params[3] # nonlinearity 
        self.beta         = params[4] # inverse temperature

    def plan_act(self):
        # retrieve memory
        obs = self.memory.sample( 'obs')[0]
        # unpack observation
        mag0, mag1 = obs
        pt = self.p_s[ 0, 0] 
        # mixture of probability and magnitude 
        vt = self.lam * (pt - ( 1 - pt)) + ( 1 - self.lam) * \
              abs( mag0 - mag1) ** self.r * np.sign( mag0 - mag1)
        # softmax action selection
        pit = 1 / ( 1 + np.exp( -self.beta * vt))
        # choice probability
        self.P_a = np.array( [ pit, 1 - pit])

class model8( model7):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.lam          = params[2] # mixture of prob and mag
        self.r            = params[3] # nonlinearity 
        self.beta         = params[4] # inverse temperature
        self.eps          = params[5] # lapse

    def plan_act(self):
        # retrieve memory
        obs = self.memory.sample( 'obs')[0]
        # unpack observation
        mag0, mag1 = obs
        pt = self.p_s[ 0, 0] 
        # mixture of probability and magnitude 
        vt = self.lam * (pt - ( 1 - pt)) + ( 1 - self.lam) * \
              abs( mag0 - mag1) ** self.r * np.sign( mag0 - mag1)
        # softmax action selection + lapse
        pit = ( 1 - self.eps) / ( 1 + np.exp( -self.beta * vt)) + self.eps / 2 
        # choice probability
        self.P_a = np.array( [ pit, 1 - pit]) 

class model11( model7):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        # stab
        self.alpha_s_stab = params[0] # learning rate of p(s) for stab task 
        self.alpha_a_stab = params[1] # learning rate of q(a) for stab task 
        self.beta_stab    = params[2] # inverse temperature for stab task 
        # vol 
        self.alpha_s_vol  = params[3] # learning rate of p(s) for vol task 
        self.alpha_a_vol  = params[4] # learning rate of q(a) for vol task 
        self.beta_vol     = params[5] # inverse temperature for vol task 
        # general
        self.lam          = params[6] # mixture of prob and mag
        self.r            = params[7] # nonlinearity 
        self.beta_a       = params[8] # inverse temperature for choice kernel

    def update( self):
        self.update_Ps()
        self.update_Pa()

    def plan_act(self):
        # retrieve memory
        ctxt, obs = self.memory.sample( 'ctxt', 'obs')
        # unpack observation
        mag0, mag1 = obs
        pt = self.p_s[ 0, 0] 
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        # mixture of probability and magnitude 
        vt = self.lam * (pt - ( 1 - pt)) + ( 1 - self.lam) * \
              abs( mag0 - mag1) ** self.r * np.sign( mag0 - mag1)
        # softmax action selection 
        pit = 1 / ( 1 + np.exp( - ( beta * vt + 
                  self.beta_a * ( self.q_a[ 0, 0] - self.q_a[ 1, 0])))) 
        # choice probability
        self.P_a = np.array( [ pit, 1 - pit]) 

class model11_new(model11):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        # stab
        self.alpha_s_stab = params[0] # learning rate of p(s) for stab task 
        self.alpha_a_stab = params[1] # learning rate of q(a) for stab task 
        self.beta_stab    = params[2] # inverse temperature for stab task 
        # vol 
        self.alpha_s_vol  = params[3] # learning rate of p(s) for vol task 
        self.alpha_a_vol  = params[4] # learning rate of q(a) for vol task 
        self.beta_vol     = params[5] # inverse temperature for vol task 
        # general
        self.lam          = params[6] # mixture of prob and mag
        self.r            = params[7] # nonlinearity 
        self.beta_a       = params[8] # inverse temperature for choice kernel
        self.w            = params[9] # trust on the model

    def plan_act(self):
        # retrieve memory
        ctxt, obs = self.memory.sample( 'ctxt', 'obs')
        # unpack observation
        mag0, mag1 = obs
        p_s = (1-self.w)*1/self.nS + self.w*self.p_s 
        pt = p_s[ 0, 0] 
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        # mixture of probability and magnitude 
        vt = self.lam * (pt - ( 1 - pt)) + ( 1 - self.lam) * \
              abs( mag0 - mag1) ** self.r * np.sign( mag0 - mag1)
        # softmax action selection 
        pit = 1 / ( 1 + np.exp( - ( beta * vt + 
                  self.beta_a * ( self.q_a[ 0, 0] - self.q_a[ 1, 0])))) 
        # choice probability
        self.P_a = np.array( [ pit, 1 - pit]) 


#==========================
#       Cog Agent
#==========================

class RDModel( Basebrain):

    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        self.alpha_s = params[0] # p(s) learning rate 
        self.alpha_a = params[1] # p(a) learning rate 
        self.beta    = params[2] # tradeoff 

    def plan_act(self):
        # construct utility function 
        u_sa = self.get_U()
        # pi(a|s) ∝ exp( βU(s,a) + log q(a))
        f_a1s   = self.beta * u_sa + np.log( self.q_a.T + eps_)
        self.pi = softmax(  f_a1s, axis=1)
        # marginal over state 
        self.P_a = ( self.p_s.T @ self.pi).reshape([-1])
    
    def update_Ps( self):
        # retrieve memory
        state = self.memory.sample( 'state')
        ## Update p_s
        # δs = It(s) - p_s
        rpe_s = self.O(state) - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += self.alpha_s * rpe_s 
    
    def update_Pa( self):
        # retrieve memory
        act = self.memory.sample( 'act')
        ## Update p_a
        # δs = It(a) - p_a
        rpe_a = self.Q(act) - self.q_a 
        # p_a = p_a + α_a * δa
        self.q_a += self.alpha_a * rpe_a 

    def update( self):
        ## Retrieve memory
        self.update_Ps()
        self.update_Pa()
    
    def pi_comp(self):
        return np.sum( self.p_s * self.pi * 
                       ( np.log( self.pi + eps_) 
                       - np.log( self.q_a.T + eps_)))
    
    def EQ( self):
        u_sa = self.get_U()
        return np.sum( self.p_s * self.pi * u_sa)

class RDModel2( RDModel):
    
    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.alpha_a_stab = params[1] # learning rate for p(a)
        self.beta_stab    = params[2] # inverse temp  
        # params for wol 
        self.alpha_s_vol  = params[3] # learning rate for p(s)
        self.alpha_a_vol  = params[4] # learning rate for p(a) 
        self.beta_vol     = params[5] # inverse temp

    def plan_act(self):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        # construct utility function 
        u_sa = self.get_U()
        # pi(a|s) ∝ exp( βU(s,a) + log q(a))
        f_a1s   = beta * u_sa + np.log( self.q_a.T + eps_)
        self.pi = softmax(  f_a1s, axis=1)
        # marginal over state 
        self.P_a = ( self.p_s.T @ self.pi).reshape([-1])
    
    def update_Ps( self):
        # retrieve memory
        ctxt, state = self.memory.sample( 'ctxt', 'state')
        # choose parameter set 
        alpha_s = self.alpha_s_stab if ctxt else self.alpha_s_vol
        ## Update p_s
        # δs = It(s) - p_s
        rpe_s = self.O(state) - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += alpha_s * rpe_s 
    
    def update_Pa( self):
        # retrieve memory
        ctxt, act = self.memory.sample( 'ctxt', 'act')
         # choose parameter set 
        alpha_a = self.alpha_a_stab if ctxt else self.alpha_a_vol
        ## Update p_a
        # δs = It(a) - p_a
        rpe_a = self.Q(act) - self.q_a 
        # p_a = p_a + α_a * δa
        self.q_a += alpha_a * rpe_a 

    def update( self):
        ## Retrieve memory
        self.update_Ps()
        self.update_Pa()

class RDModel2_exp( RDModel2):

    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.alpha_a_stab = params[1] # learning rate for p(a)
        self.beta_stab    = params[2] # inverse temp  
        # params for wol 
        self.alpha_s_vol  = params[3] # learning rate for p(s)
        self.alpha_a_vol  = params[4] # learning rate for p(a) 
        self.beta_vol     = params[5] # inverse temp
        # general 
        self.beta         = params[6]

    def plan_act(self):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        # construct utility function 
        u_sa = self.get_U()
        # pi(a|s) ∝ exp( βU(s,a) + log q(a))
        f_a1s   = beta * u_sa + np.log( self.q_a.T + eps_)
        self.pi = softmax( f_a1s, axis=1)
        # marginal over state 
        self.P_a = (softmax( self.beta * self.p_s.T @ self.pi, 
                            axis=1)).reshape([-1])

class RDModel2_exp2( RDModel2):

    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.alpha_a_stab = params[1] # learning rate for p(a)
        self.beta_stab    = params[2] # inverse temp  
        # params for wol 
        self.alpha_s_vol  = params[3] # learning rate for p(s)
        self.alpha_a_vol  = params[4] # learning rate for p(a) 
        self.beta_vol     = params[5] # inverse temp
        # general 
        self.beta         = params[6]

    def plan_act(self):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        # construct utility function 
        u_sa = self.get_U()
        # pi(a|s) ∝ exp( βU(s,a) + log q(a))
        f_a1s   = beta * u_sa + np.log( self.q_a.T + eps_)
        f_a1s[ 0, 1] = 0
        f_a1s[ 1, 0] = 0 
        self.pi = softmax( f_a1s, axis=1)
        # marginal over state 
        self.P_a = (softmax( self.beta * self.p_s.T @ self.pi, 
                            axis=1)).reshape([-1])

class RDModel3( RDModel2):
    
    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.alpha_a_stab = params[1] # learning rate for p(a)
        self.beta_stab    = params[2] # inverse temp  
        self.tau_stab     = params[3] # trust on model 
        # params for wol 
        self.alpha_s_vol  = params[4] # learning rate for p(s)
        self.alpha_a_vol  = params[5] # learning rate for p(a) 
        self.beta_vol     = params[6] # inverse temp
        self.tau_vol      = params[7] # trust on model

    def plan_act(self):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        tau  = self.tau_stab if ctxt else self.tau_vol
        # construct utility function 
        u_sa = self.get_U()
        # pi(a|s) ∝ exp( βU(s,a) + log q(a))
        f_a1s   = beta * u_sa + tau * np.log( self.q_a.T + eps_)
        self.pi = softmax(  f_a1s, axis=1)
        # marginal over state 
        self.P_a = ( self.p_s.T @ self.pi).reshape([-1])

class SMModel( Basebrain):
    
    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.beta_stab    = params[1] # inverse temp  
        # params for wol 
        self.alpha_s_vol  = params[2] # learning rate for p(s)
        self.beta_vol     = params[3] # inverse temp

    def plan_act(self):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        # construct utility function 
        u_sa = self.get_U()
        # pi(a|s) ∝ exp( βU(s,a))
        f_a1s   = beta * u_sa 
        self.pi = softmax( f_a1s, axis=1)
        # marginal over state 
        self.P_a = ( self.p_s.T @ self.pi).reshape([-1])
    
    def update_Ps( self):
        # retrieve memory
        ctxt, state = self.memory.sample( 'ctxt', 'state')
        # choose parameter set 
        alpha_s = self.alpha_s_stab if ctxt else self.alpha_s_vol
        ## Update p_s
        # δs = It(s) - p_s
        rpe_s = self.O(state) - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += alpha_s * rpe_s 

    def update( self):
        ## Retrieve memory
        self.update_Ps()

class SMModel2( SMModel):

    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.beta_stab    = params[1] # inverse temp  
        # params for wol 
        self.alpha_s_vol  = params[2] # learning rate for p(s)
        self.beta_vol     = params[3] # inverse temp
        # general
        self.r            = params[4] # nonlinearity 

    def plan_act(self):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        # construct utility function 
        u_sa = self.get_U()
        # pi(a|s) ∝ exp( βU(s,a))
        f_a1s   = beta * (u_sa ** self.r) 
        self.pi = softmax( f_a1s, axis=1)
        # marginal over state 
        self.P_a = ( self.p_s.T @ self.pi).reshape([-1])

class NM( Basebrain):
    '''Normative model
    '''
    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.k            = params[2] # capacity

    def plan_act( self):
        # construct utility function 
        u_sa = self.get_U()
        # derive the optimal policy
        self.pi, p_a, _, _ = RD(u_sa, self.p_s, self.k)
        # marginal over state 
        self.p_a1m = p_a[ :, 0]

class NMa( NM):
    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params( self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.alpha_a_stab = params[1] # learning rate for p(a) 
        # params for wol 
        self.alpha_s_vol  = params[2] # learning rate for p(s)
        self.alpha_a_vol  = params[3] # learning rate for p(a) 
        # general params
        self.k            = params[4] # capacity 

    def plan_act( self):
        # construct utility function 
        u_sa = self.get_U()
        # derive the optimal policy
        self.pi, p_a, _, _ = RD( u_sa, self.p_s, self.k,)
        # marginal over state 
        self.p_a1m = p_a[ :, 0]
        
    def update( self):
        ## Retrieve memory
        self.update_Ps()
        self.update_Pa()

class TM( Basebrain):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.tau_stab     = params[1] # inverse temp 
        # params for vol 
        self.alpha_s_vol  = params[2] # learning rate for p(s)
        self.tau_vol      = params[3] # inverse temp 
    
    def plan_act(self,):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = 1 / self.tau_stab if ctxt else 1 / self.tau_vol
        # construct utility function 
        u_sa = self.get_U()
        # unpack observation 
        self.pi = softmax( beta * u_sa + np.log( self.p_a.T + eps_), axis=1)
        self.p_a = self.pi.T @ self.p_s  
        # choice probability
        self.p_a1m = self.p_a[ :, 0]

class TMa( Basebrain):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.alpha_a_stab = params[1] # learning rate for p(a)
        self.tau_stab     = params[2] # inverse temp  
        # params for wol 
        self.alpha_s_vol  = params[3] # learning rate for p(s)
        self.alpha_a_vol  = params[4] # learning rate for p(a) 
        self.tau_vol      = params[5] # inverse temp

    def plan_act(self,):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = 1 / self.tau_stab if ctxt else 1 / self.tau_vol
        # construct utility function 
        u_sa = self.get_U()
        # unpack observation 
        self.pi = softmax( beta * u_sa + np.log( self.p_a.T + eps_), axis=1)
        p_a = self.pi.T @ self.p_s  
        # choice probability
        self.p_a1m = p_a[ :, 0]

class SM( Basebrain):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.alpha_t_stab = params[1] # learning rate for τ
        # params for wol 
        self.alpha_s_vol  = params[2] # learning rate for p(s)
        self.alpha_t_vol  = params[3] # learning rate for τ
        # tau 
        self.k            = params[4] # capacity 
        self.tau          = params[5] # tradeoff 0 

    def plan_act(self,):
        # choose parameter set 
        beta = 1 / self.tau 
        # construct utility function 
        u_sa = self.get_U()
        # unpack observation 
        self.pi = softmax( beta * u_sa + np.log( self.p_a.T + eps_), axis=1)
        p_a = self.pi.T @ self.p_s  
        # choice probability
        self.p_a1m = p_a[ :, 0]

    def update_tau( self):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        alpha_t = self.alpha_t_stab if ctxt else self.alpha_t_vol
        p_a = self.pi.T @ self.p_s
        vio = I( self.p_s, self.pi, p_a) - self.k
        tau = self.tau + alpha_t * vio 
        self.tau = np.clip( tau, 1e-3, 1000)

    def update( self):
        ## Retrieve memory
        self.update_Ps()
        self.update_tau()

class SMa( SM):

    def _load_free_params(self, params):
        # params for stab
        self.alpha_s_stab = params[0] # learning rate for p(s)
        self.alpha_a_stab = params[1]
        self.alpha_t_stab = params[2] # learning rate for τ
        # params for vol 
        self.alpha_s_vol  = params[3] # learning rate for p(s)
        self.alpha_a_vol  = params[4] # learning rate for τ
        self.alpha_t_vol  = params[5] 
        # tau 
        self.k            = params[6] # capacity 
        self.tau          = params[7] # tradeoff 0 

    def update( self):
        ## Retrieve memory
        self.update_Ps()
        self.update_Pa()
        self.update_tau()

#--------------------------
#        Bayes Agent
#--------------------------

def rbeta( r, v):
    '''Reparameterized beta
    r = a / (a+b)
    v = -log(a+b)
    '''
    a = r*np.exp(-v)
    b = np.exp(-v)*(1-r)
    return beta( a, b)

class BayesLearner( Basebrain):

    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
        self._init_dists()

    def _load_free_params( self, params):
        self.beta_stab   = params[0] # stable beta 
        self.beta_vol    = params[1] # vol beta 

    def _init_dists(self):
        self._discretize()
        self.S, self.A = {}, {} 
        dists_name = ['p_V1VK', 'p_R1VR', 'delta']
        for d in [self.S, self.A]:
            for dist in dists_name:
                d[dist] = eval( f'self._init_{dist}')()

    def _discretize( self,):
        '''Discretize all variables
            r: the expectation of bernoulli dist 
            v: indicates the variation of bernoulli ,
            k: the variance of volatitity
            Default dim convention:
            dim: yt, vt, rt, vt-1, rt-1, k
        '''
        # get discerete space 
        self.n_split = 30 
        self.r_space = np.linspace( .01, .99, self.n_split)
        self.v_space = np.linspace( -11,  -2, self.n_split)
        self.k_space = np.linspace(  -2,   2, self.n_split)

    def _init_p_V1VK( self,):
        '''p(Vt|Vt-1=i,k) = N( i, exp(k))
        '''
        p_V1VK = np.zeros( [ self.n_split, self.n_split, self.n_split])
        for i, vi in enumerate( self.v_space):
            for k, kk in enumerate( self.k_space):
                p_V1VK[ :, i, k] = norm.pdf( self.v_space, loc=vi, scale=np.exp(kk))
        p_V1VK /= p_V1VK.sum(0,keepdims=True)
        return p_V1VK

    def _init_p_R1VR( self):
        '''p(Rt|Vt-1=i,Rt-1=j) = rBeta(j,i)
        '''
        p_R1VR = np.zeros( [ self.n_split, self.n_split, self.n_split])
        for j, rj in enumerate( self.r_space):
            for i, vi in enumerate( self.v_space):
                p_R1VR[ :, i, j] = rbeta( rj, vi).pdf(self.r_space)
        p_R1VR /= p_R1VR.sum(0,keepdims=True)
        return p_R1VR

    def _init_delta( self,):
        '''δ0(vi,rj,kk)
            Init with Perks prior 
        '''
        f_VRK =  np.ones( [ self.n_split, self.n_split, self.n_split])
        return f_VRK / f_VRK.sum()

    def plan_act(self):
        # retrieve memory
        ctxt = self.memory.sample( 'ctxt')[0]
        # choose parameter set 
        beta = self.beta_stab if ctxt else self.beta_vol
        # construct utility function 
        u_sa = self.get_U()
        # pi(a|s) ∝ exp( βU(s,a) + log q(a))
        f_a1s   = beta * u_sa + np.log( self.q_a.T + eps_)
        self.pi = softmax(  f_a1s, axis=1)
        # marginal over state 
        self.P_a = ( self.p_s.T @ self.pi).reshape([-1])

    def Bayes_update(self, var_, y):
        '''update δ with y 
        '''
        dist = eval(f'self.{var_}')
        # get p(yt|rt): dim: rt
        p_y1r = bernoulli.pmf( y, self.r_space)
        # ∑i p(vt|vt-1=i, k) δ(i,rt-1,k)
        # dim: vt vt-1 @ vt-1, rt-1 --> vt, rt-1
        delta1 = np.zeros( [self.n_split]*3) 
        for k in range(self.n_split):
            delta1[:,:,k] = dist['p_V1VK'][:,:,k] @ dist['delta'][:,:,k]
        # ∑j p(rt|vt, rt-1=j) δ(vt,j,k)
        # dim:  rt rt-1 @ rt-1, k = rt k 
        delta2 = np.zeros( [self.n_split]*3) 
        for i in range(self.n_split):
            delta2[i,:,:] = dist['p_R1VR'][:,i,:]@delta1[i,:,:]
        # δ(vt,j,k) * p(y|rt=j)
        # get new delta: vt, rt, k
        delta = p_y1r[ np.newaxis, :, np.newaxis] * delta2
        dist['delta'] = delta / delta.sum()
        # get some prediction
        return (dist['delta'].sum(axis=(0,2))*self.r_space).sum()
        
    def update_Ps( self):
        # retrieve memory
        state = self.memory.sample( 'state')[0]
        prob_s = self.Bayes_update( 'S', 1-state)
        self.p_s = np.array( [[ prob_s, 1-prob_s]]).T #nSx1
    
    def update_Pa( self):
        # retrieve memory
        act = self.memory.sample( 'act')[0]
        prob_a = self.Bayes_update( 'A', 1-act)
        self.p_a = np.array( [[ prob_a, 1-prob_a]]).T #nAx1

    def update( self):
        ## Retrieve memory
        self.update_Ps()
        self.update_Pa()
    
    def pi_comp(self):
        return np.sum( self.p_s * self.pi * 
                       ( np.log( self.pi + eps_) 
                       - np.log( self.q_a.T + eps_)))
    
    def EQ( self):
        u_sa = self.get_U()
        return np.sum( self.p_s * self.pi * u_sa)

class BayesNoPolicy( BayesLearner):

    def __init__( self, nS, nA, rng, params=[]):
        super().__init__( nS, nA, rng)
        if len( params):
            self._load_free_params( params)
        self._init_dists()

    def _load_free_params( self, params):
        self.beta_stab   = params[0] # stable beta 

    def plan_act(self):
        self.P_a = (self.p_s).reshape([-1])