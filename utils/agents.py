import numpy as np
from numpy.core.fromnumeric import argmax 
from scipy.special import softmax, logsumexp 

# get the machine epsilon
eps_ = 1e-12
max_ = 1e+10

# the replay buffer to store the memory 
class simpleBuffer:
    '''Simple Buffer 2.0

    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__( self):
        self.m = [] 
        
    def push( self, m_dict):
        self.m = m_dict 
        
    def sample( self, *args):
        lst = []
        for key in args:
            lst.append( self.m[ key])
        return lst

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Base agent class    %
%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''
class Basebrain:
    
    def __init__( self, state_dim, act_dim, rng):
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.rng        = rng 
        self.act_space  = range( self.act_dim)
        self._init_critic()
        self._init_actor()
        self._init_marg_state()
        self._init_marg_action()
        self._init_memory()

    def _init_marg_state( self):
        self.p_s = np.ones( [self.state_dim, 1]
                          ) * 1 / self.state_dim

    def _init_marg_action( self):
        self.p_a = np.ones( [self.act_dim, 1]
                          ) * 1 / self.act_dim

    def _init_choice_prob( self):
        self.p_a1x = np.ones( [self.act_dim,]
                          ) * 1 / self.act_dim

    def _init_memory( self):
        self.memory = simpleBuffer()
        
    def _init_critic( self):
        self.q_s     = np.ones( [ 1, self.state_dim]) / self.state_dim
        self.q_table = np.ones( [self.state_dim, self.act_dim]
                          ) * 1 / self.act_dim

    def _init_actor( self):
        self.pi = np.ones( [ self.state_dim, self.act_dim]
                          ) * 1 / self.act_dim
    
    def plan_act( self, obs):
        '''Generate action given observation
            p(a|xt)
        '''
        return NotImplementedError
        
    def get_act( self):
        '''Sample from p(a|xt)
        '''
        return self.rng.choice( self.act_space, p=self.p_a1x)
        
    def eval_act( self, act):
        '''get from p(at|xt)
        '''
        return self.p_a1x[ act]
        
    def update( self):
        '''Learning 
        '''
        return NotImplementedError
    
    def pi_comp( self):
        return None 

    def EQ( self, obs):
        return None 

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Models in the paper     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
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

    def update( self):
        
        ## Retrieve memory
        ctxt, state = self.memory.sample( 'ctxt', 'state')

        # choose ctxt
        if ctxt: 
            alpha_s = self.alpha_s_stab
        else:
            alpha_s = self.alpha_s_vol

        ## Update p_s
        # calculate δs = It(s) - p_s
        I_st = np.zeros( [ self.state_dim, 1])
        I_st[ state, 0] = 1.
        rpe_s = I_st - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += alpha_s * rpe_s 

    def plan_act( self, obs):
        
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
        self.p_a1x = np.array( [ pit, 1 - pit])

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

    def plan_act(self, obs):
        
        # unpack observation
        mag0, mag1 = obs
        pt = self.p_s[ 0, 0] 
        # mixture of probability and magnitude 
        vt = self.lam * ( pt - ( 1 - pt)) + \
             ( 1 - self.lam) * ( mag0 - mag1)
        # softmax action selection
        pit = 1 / ( 1 + np.exp( -self.beta * vt))
        # choice probability
        self.p_a1x = np.array( [ pit, 1 - pit])

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

    def plan_act(self, obs):
        
        # unpack observation
        mag0, mag1 = obs
        pt = self.p_s[ 0, 0] 
        # mixture of probability and magnitude 
        vt = self.lam * (pt - ( 1 - pt)) + ( 1 - self.lam) * \
              abs( mag0 - mag1) ** self.r * np.sign( mag0 - mag1)
        # softmax action selection
        pit = 1 / ( 1 + np.exp( -self.beta * vt))
        # choice probability
        self.p_a1x = np.array( [ pit, 1 - pit])

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

    def plan_act(self, obs):
        
        # unpack observation
        mag0, mag1 = obs
        pt = self.p_s[ 0, 0] 
        # mixture of probability and magnitude 
        vt = self.lam * (pt - ( 1 - pt)) + ( 1 - self.lam) * \
              abs( mag0 - mag1) ** self.r * np.sign( mag0 - mag1)
        # softmax action selection + lapse
        pit = ( 1 - self.eps) / ( 1 + np.exp( -self.beta * vt)) + self.eps / 2 
        # choice probability
        self.p_a1x = np.array( [ pit, 1 - pit]) 

class model11( model7):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.lam          = params[2] # mixture of prob and mag
        self.r            = params[3] # nonlinearity 
        self.alpha_a      = params[4] # learning rate of choice kernel
        self.beta         = params[5] # inverse temperature
        self.beta_a       = params[6] # inverse temperature for choice kernel

    def update( self):

        ## Retrieve memory
        ctxt, state, action = self.memory.sample( 'ctxt', 'state', 'action')

        # choose ctxt
        if ctxt: 
            alpha_s = self.alpha_s_stab
        else:
            alpha_s = self.alpha_s_vol

        ## Update p_s
        # calculate δs = It(s) - p_s
        I_st = np.zeros( [ self.state_dim, 1])
        I_st[ state, 0] = 1.
        rpe_s = I_st - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += alpha_s * rpe_s 

        ## Update p_a
        # calculate δa = It(a) - p_a
        I_at = np.zeros( [ self.act_dim, 1])
        I_at[ action, 0] = 1.
        rpe_a = I_at - self.p_a 
        # p_a = p_a + α_a * δa
        self.p_a += self.alpha_a * rpe_a 

    def plan_act(self, obs):
        
        # unpack observation
        mag0, mag1 = obs
        pt = self.p_s[ 0, 0] 
        # mixture of probability and magnitude 
        vt = self.lam * (pt - ( 1 - pt)) + ( 1 - self.lam) * \
              abs( mag0 - mag1) ** self.r * np.sign( mag0 - mag1)
        # softmax action selection 
        pit = 1 / ( 1 + np.exp( - ( self.beta * vt + 
                  self.beta_a * ( self.p_a[ 0, 0] - self.p_a[ 1, 0])))) 
        # choice probability
        self.p_a1x = np.array( [ pit, 1 - pit]) 

#========================================
#     Eligibility trace for state       
#========================================   

class model11_e( model11):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        self.alpha     = params[0] # learning rate for the state in the stable task 
        self.nu        = params[1] # the decay of eligibility 
        self.lam       = params[2] # mixture of prob and mag
        self.r         = params[3] # nonlinearity 
        self.alpha_a   = params[4] # learning rate of choice kernel
        self.beta      = params[5] # inverse temperature
        self.beta_a    = params[6] # inverse temperature for choice kernel
        self.e_trace   = 0

    def update( self):

        ## Retrieve memory
        state, action = self.memory.sample( 'state', 'action')

        ## Update p_s
        # calculate δs = It(s) - p_s
        I_st = np.zeros( [ self.state_dim, 1])
        I_st[ state, 0] = 1.
        self.e_trace = self.nu * self.e_trace + I_st
        self.e_trace /= self.e_trace.sum()
        rpe_s = self.e_trace - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += self.alpha * rpe_s 

        ## Update p_a
        # calculate δa = It(a) - p_a
        I_at = np.zeros( [ self.act_dim, 1])
        I_at[ action, 0] = 1.
        rpe_a = I_at - self.p_a 
        # p_a = p_a + α_a * δa
        self.p_a += self.alpha_a * rpe_a 

class model11_m( model11):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        self.alpha     = params[0] # learning rate for the state in the stable task 
        self.lam       = params[1] # mixture of prob and mag
        self.r         = params[2] # nonlinearity 
        self.alpha_a   = params[3] # learning rate of choice kernel
        self.beta      = params[4] # inverse temperature
        self.beta_a    = params[5] # inverse temperature for choice kernel

    def update( self):

        ## Retrieve memory
        state, action = self.memory.sample( 'state', 'action')

        ## Update p_s
        # calculate δs = It(s) - p_s
        I_st = np.zeros( [ self.state_dim, 1])
        I_st[ state, 0] = 1.
        rpe_s = I_st - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += self.alpha * rpe_s 

        ## Update p_a
        # calculate δa = It(a) - p_a
        I_at = np.zeros( [ self.act_dim, 1])
        I_at[ action, 0] = 1.
        rpe_a = I_at - self.p_a 
        # p_a = p_a + α_a * δa
        self.p_a += self.alpha_a * rpe_a 

#==========================================
#     Resource rational model for policy      
#===========================================  

class RRmodel( model11):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params( self, params):
        self.alpha_s  = params[0] # learning rate for the state in the stable task 
        self.alpha_a  = params[1] # learning rate of choice kernel
        self.beta     = params[2] # temperature
        self.nu       = 0
        self.e_trace  = 0 

    def update( self):

        ## Retrieve memory
        state, action = self.memory.sample( 'state', 'action')

        ## Update p_s
        # calculate δs = 1 - v(st)
        I_st = np.zeros( [ self.state_dim, 1])
        I_st[ state, 0] = 1.
        self.e_trace = self.nu * self.e_trace + I_st
        self.e_trace /= self.e_trace.sum()
        rpe_s = self.e_trace - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += self.alpha_s * rpe_s 

        ## Update p_a
        # calculate δa = It(a) - p_a
        I_at = np.zeros( [ self.act_dim, 1])
        I_at[ action, 0] = 1.
        rpe_a = I_at - self.p_a 
        # p_a = p_a + α_a * δa
        self.p_a += self.alpha_a * rpe_a 
    
    def plan_act( self, obs):

        # get Q
        mag0, mag1 = obs 
        Q = np.array([[ mag0,    0],
                      [    0, mag1]])
        # get π ∝ exp[ βQ(s,a) + log p_a(a)]
        log_pi = self.beta * Q + np.log( self.p_a.T + eps_) #sa
        self.pi = np.exp( log_pi - logsumexp( 
                          log_pi, keepdims=True, axis=1)) #sa
        # action probability given observation
        self.p_a1x = ( self.p_s.T @ self.pi).reshape([-1]) 

    def log_p( self, x):
        log_p = np.log( x)
        log_p[ log_p <= -1e11 ] = 0.
        return log_p 

    def pi_comp( self):
        MI = np.sum(self.p_s * self.pi * (self.log_p( self.pi)
                     - self.log_p( self.p_a.T)))
        return MI

    def EQ( self, obs):
        # get Q
        mag0, mag1 = obs 
        Q = np.array([[ mag0,    0],
                      [    0, mag1]])
        # EQ = ∑_s ∑π(a|s) Q(s,a)
        return np.sum( self.p_s * self.pi * Q)

class RRmodel_e( RRmodel):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params( self, params):
        self.alpha_s  = params[0] # learning rate for the state in the stable task 
        self.alpha_a  = params[1] # learning rate of choice kernel
        self.beta     = params[2] # temperature
        self.nu       = params[3]
        self.e_trace  = 0

class RRmodel_ctxt( RRmodel):

    def __init__( self, state_dim, act_dim, rng, params=[]):
        super().__init__( state_dim, act_dim, rng)
        if len( params):
            self._load_free_params( params)

    def _load_free_params( self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.alpha_a_stab = params[2] # learning rate of choice kernel stab
        self.alpha_a_vol  = params[3] # learning rate of choice kernel volatile
        self.beta_stab    = params[4] # temperature stable
        self.beta_vol     = params[5] # temperature volatile 

    def update( self):

        ## Retrieve memory
        ctxt, state, action = self.memory.sample( 'ctxt', 'state', 'action')

        # choose ctxt
        alpha_s = self.alpha_s_stab if ctxt else self.alpha_s_vol
        alpha_a = self.alpha_a_stab if ctxt else self.alpha_a_vol

        ## Update p_s
        # calculate δs = 1 - v(st)
        I_st = np.zeros( [ self.state_dim, 1])
        I_st[ state, 0] = 1.
        rpe_s = I_st - self.p_s 
        # p_s = p_s + α_s * δs
        self.p_s += alpha_s * rpe_s

        ## Update p_a
        # calculate δa = It(a) - p_a
        I_at = np.zeros( [ self.act_dim, 1])
        I_at[ action, 0] = 1.
        rpe_a = I_at - self.p_a 
        # p_a = p_a + α_a * δa
        self.p_a += alpha_a * rpe_a 
    
    def plan_act( self, obs):
        # get context 
        ctxt = self.memory.sample('ctxt')
        beta = self.beta_stab if ctxt else self.beta_vol
        # get Q
        mag0, mag1 = obs 
        Q = np.array([[ mag0,    0],
                      [    0, mag1]])
        # get π ∝ exp[ βQ(s,a) + log p_a(a)]
        log_pi = beta * Q + np.log( self.p_a.T + eps_) #sa
        self.pi = np.exp( log_pi - logsumexp( 
                          log_pi, keepdims=True, axis=1)) #sa
        # action probability given observation
        self.p_a1x = ( self.p_s.T @ self.pi).reshape([-1]) 




