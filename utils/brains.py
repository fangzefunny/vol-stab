import numpy as np 
from scipy.special import softmax, logsumexp 

# get the machine epsilon
eps_ = 1e-10
max_ = 1e+10

# the replay buffer to store the memory 
class simpleBuffer:
    
    def __init__( self):
        self.lst = []
        
    def push( self, *args):
        self.lst = tuple([ x for x in args]) 
        
    def sample( self ):
        return self.lst

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Base agent class    %
%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''

class Basebrain:
    
    def __init__( self, state_dim, act_dim):
        self.state_dim  = state_dim
        self.act_dim    = act_dim
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
        self.q_table = np.ones( [self.state_dim, self.act_dim]
                          ) * 1 / self.act_dim

    def _init_actor( self):
        self.pi = np.ones( [ self.state_dim, self.act_dim]
                          ) * 1 / self.act_dim
    
    def plan_act( self, obs):
        return NotImplementedError
        
    def get_act( self):
        return np.random.choice( self.act_space, p=self.p_a1x)
        
    def eval_act( self, act):
        return self.p_a1x[ act]
        
    def update( self):
        return NotImplementedError

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Models in the paper     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

class model1( Basebrain):

    def __init__( self, state_dim, act_dim, params=[]):
        super().__init__( state_dim, act_dim)
        if len( params):
            self._load_free_params( params)
    
    def _load_free_params( self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.gamma        = params[2] # risk preference
        self.beta         = params[3] # inverse temperature

    def update( self):
        
        ## Retrieve memory
        ctxt, _, _, state, _, _ = self.memory.sample()

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

    def __init__( self, state_dim, act_dim, params=[]):
        super().__init__( state_dim, act_dim)
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

    def __init__( self, state_dim, act_dim, params=[]):
        super().__init__( state_dim, act_dim)
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

    def __init__( self, state_dim, act_dim, params=[]):
        super().__init__( state_dim, act_dim)
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

    def __init__( self, state_dim, act_dim, params=[]):
        super().__init__( state_dim, act_dim)
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
        ctxt, _, action, state, _, _ = self.memory.sample()

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

class model11_m( model11):

    def __init__( self, state_dim, act_dim, params=[]):
        super().__init__( state_dim, act_dim)
        if len( params):
            self._load_free_params( params)

    def _load_free_params(self, params):
        self.alpha_s_stab = 0 # learning rate for the state in the stable task 
        self.alpha_s_vol  = 0 # learning rate for the state in the volatile task 
        self.lam          = params[0] # mixture of prob and mag
        self.r            = params[1] # nonlinearity 
        self.alpha_a      = params[2] # learning rate of choice kernel
        self.beta         = params[3] # inverse temperature
        self.beta_a       = params[4] # inverse temperature for choice kernel

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Model Extension     %
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''
class RRmodel( model11):

    def __init__( self, state_dim, act_dim, params=[]):
        super().__init__( state_dim, act_dim)
        if len( params):
            self._load_free_params( params)

    def _load_free_params( self, params):
        self.alpha_s_stab = params[0] # learning rate for the state in the stable task 
        self.alpha_s_vol  = params[1] # learning rate for the state in the volatile task 
        self.alpha_a      = params[2] # learning rate of choice kernel
        self.beta         = params[3] # temperature
    
    def plan_act( self, obs):

        # get Q
        mag0, mag1 = obs 
        Q = np.array([[ mag0,    0],
                      [    0, mag1]])
        # get π ∝ exp[ βQ(s,a) + log p_a(a)]
        log_pi = self.beta * Q + np.log( self.p_a.T + eps_) #sa
        self.pi = np.exp( log_pi - logsumexp( 
                          log_pi, keepdims=True, axis=1)) #sa

        self.p_a1x = (self.p_s.T @ self.pi).reshape([-1]) 

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

class RRmodel1( RRmodel):

    def __init__( self, state_dim, act_dim, params=[]):
        super().__init__( state_dim, act_dim)
        if len( params):
            self._load_free_params( params)

    def _load_free_params( self, params):
        self.alpha_s_stab = params[0] # learning rate of state in the stable block
        self.alpha_s_vol  = params[1] # learning rate of state in the volatile block 
        self.alpha_pi     = params[2] # learning rate of the policy 
        self.alpha_a      = params[3] # learning rate of choice kernel
        self.beta         = params[4] # temperature
    
    def plan_act( self, obs):
        # get Q
        mag0, mag1 = obs 
        Q = np.array([[ mag0,    0],
                      [    0, mag1]])
        # get π ∝ exp[ βQ(s,a) + log p_a(a)]
        log_pi = self.beta * Q + np.log( self.p_a.T + eps_) #sa
        pi_target = np.exp( log_pi - logsumexp( 
                          log_pi, keepdims=True, axis=1)) #sa
        self.pi += self.alpha_pi * (pi_target - self.pi)

        self.p_a1x = (self.p_s.T @ self.pi).reshape([-1]) 



    

        





