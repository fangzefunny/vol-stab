import numpy as np
from scipy.special import softmax

# get the machine epsilon
eps_ = 1e-12
max_ = 1e+12

def MI(p_s, pi, q_a):
    return np.sum(p_s * pi * (np.log(pi + eps_) - np.log(q_a.T + eps_)))

# ---------  Replay Buffer ----------- #

class simpleBuffer:
    '''Simple Buffer 2.0
    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__(self):
        self.m = {}
        
    def push(self, m_dict):
        self.m = { k: m_dict[k] for k in m_dict.keys()}
        
    def sample(self, *args):
        lst = [self.m[k] for k in args]
        if len(lst)==1: return lst[0]
        else: return lst

# ---------  Base ----------- #

class baseAgent:
    '''Base Agent'''
    name     = 'base'
    n_params = 0
    bnds     = []
    pbnds    = []
    p_name   = []  
    n_params = 0 
    # value of interest, used for output
    # the interesting variable in simulation
    voi      = []
    
    def __init__(self, nA, params):
        self.nA = nA 
        self.load_params(params)
        self._init_Believes()
        self._init_Buffer()

    def load_params(self, params): 
        return NotImplementedError

    def _init_Buffer(self):
        self.buffer = simpleBuffer()
    
    def _init_Believes(self):
        self._init_Critic()
        self._init_Actor()
        self._init_Dists()

    def _init_Critic(self): pass 

    def _init_Actor(self): pass 

    def _init_Dists(self):  pass

    def learn(self): 
        return NotImplementedError

    def _policy(self): 
        return NotImplementedError

    def control(self, a, rng=None, mode='sample'):
        '''control problem 
            mode: -'get': get an action, need a rng 
                  -'eval': evaluate the log like 
        '''
        p_A = self._policy() 
        if mode == 'eval': 
            return np.log(p_A[a]+eps_)
        elif mode == 'sample': 
            return rng.choice(range(self.nA), p=p_A), np.log(p_A[a]+eps_)

# ---------  Gagne RL ----------- #

class gagRL(baseAgent):
    name     = 'Gagne RL'
    bnds     = [(0, 1), (0, 1), (0, 30)]
    pbnds    = [(0,.5), (0,.5), (0, 10)]
    p_name   = ['α_STA', 'α_VOL', 'β']  
    n_params = len(bnds)
    voi      = ['p'] 
   
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.beta      = params[2]

    def _init_Critic(self):
        self.p     = 1/2
        self.p_S   = np.array([1-self.p, self.p]) 

    def learn(self):
        self._learnCritic()

    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt','state')
        alpha = self.alpha_sta if c=='stable' else self.alpha_vol
        self.p += alpha * (o - self.p)
        self.p_S = np.array([1-self.p, self.p])

    def _policy(self):
        m1, m2 = self.buffer.sample('mag0','mag1')
        mag = np.array([m1, m2])
        return softmax(self.beta * self.p_S * mag)

    def print_p(self):
        return (1-self.p)

class gagModel(gagRL):
    name     = 'Gagne best model'
    bnds     = [(0, 1), (0, 1), (0, 30), (0, 30), 
                (0, 1), (0, 30), (0, 1), (0, 1)]
    pbnds    = [(0,.5), (0,.5), (0, 10), (0, 10), 
                (0, 1), (0, 30), (0, 1), (0, 1)]
    p_name   = ['α_STA', 'α_VOL', 'β_STA', 'β_VOL', 
                'α_ACT', 'β_ACT', 'λ', 'r']  
    n_params = len(bnds)
    voi      = ['p'] 
   
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.beta_sta  = params[2]
        self.beta_vol  = params[3]
        self.alpha_act = params[4]
        self.beta_act  = params[5]
        self.lamb      = params[6]
        self.r         = params[7]
    
    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _init_Actor(self):
        self.q     = 1/2
       
    def _learnActor(self):
        a = self.buffer.sample('act')
        self.q += self.alpha_act * (a - self.q)
       
    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        beta = self.beta_sta if c=='stable' else self.beta_vol
        v    = self.lamb*(self.p - (1-self.p)) \
               + (1-self.lamb)*abs(m1-m0)**self.r*np.sign(m1-m0)
        va   = beta*v + self.beta_act*(self.q - (1-self.q))
        pa   = 1 / (1 + np.exp(-va))
        return np.array([1-pa, pa])

class gagModel2(gagModel):
    name     = 'Gagne best model'
    bnds     = [(0, 1), (0, 1), (0, 30), (0, 30), 
                (0, 1), (0, 1)]
    pbnds    = [(0,.5), (0,.5), (0, 10), (0, 10), 
                (0, 1), (0, 1)]
    p_name   = ['α_STA', 'α_VOL', 'β_STA', 'β_VOL', 
                'λ', 'r']  
    n_params = len(bnds)
    voi      = ['p'] 
   
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.beta_sta  = params[2]
        self.beta_vol  = params[3]
        self.lamb      = params[4]
        self.r         = params[5]
    
    def learn(self):
        self._learnCritic()
       
    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        beta = self.beta_sta if c=='stable' else self.beta_vol
        v    = self.lamb*(self.p - (1-self.p)) \
               + (1-self.lamb)*abs(m1-m0)**self.r*np.sign(m1-m0)
        va   = beta*v 
        pa   = 1 / (1 + np.exp(-va))
        return np.array([1-pa, pa])

        
# ---------  min CE ----------- #

sigmoid = lambda x: 1 / (1+np.exp(-x))

class ceModel(gagRL):
    name     = 'ce RL'
    bnds     = [(0,30), (0,30), (0, 30)]
    pbnds    = [(0, 2), (0, 2), (0, 10)]
    p_name   = ['α_STA', 'α_VOL', 'β']
    n_params = len(bnds)
    voi      = ['p'] 

    def _init_Critic(self):
        self.theta = 0 
        self.p     = sigmoid(self.theta)
        self.p_S   = np.array([1-self.p, self.p]) 

    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt','state')
        alpha = self.alpha_sta if c=='stable' else self.alpha_vol
        self.theta += alpha * (o - self.p)
        self.p = sigmoid(self.theta)
        self.p_S = np.array([1-self.p, self.p])

class CSCE(ceModel):
    name     = 'ce RL'
    bnds     = [(0,50), (0,50), (0, 50), (0, 50), (0,50), (0, 50)]
    pbnds    = [(0, 2), (0, 2), (0, 10), (0, 10), (0, 3), (0, 10)]
    p_name   = ['α_STA', 'α_VOL', 'λ_STA', 'λ_VOL', 'α_ACT', 'β',]
    n_params = len(bnds)
    voi      = ['p'] 

    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.lamb_sta  = params[2]
        self.lamb_vol  = params[3]
        self.alpha_act = params[4]
        self.beta      = params[5]

    def _init_Actor(self):
        self.phi = 0 
        self.q   = sigmoid(self.phi)
        self.q_A = np.array([1-self.q, self.q]) 

    def _learnActor(self):
        a = self.buffer.sample('act')
        self.phi += self.alpha_act * (a - self.q)
        self.q = sigmoid(self.phi)
        self.q_A = np.array([1-self.q, self.q])
    
    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        lamb = self.lamb_sta if c=='stable' else self.lamb_vol
        mag = np.array([m0, m1])
        u_A = self.p_S*mag 
        return softmax(self.beta*u_A + lamb*np.log(self.q_A+eps_))

class mix(CSCE):
    name     = 'ce RL'
    bnds     = [(0,50), (0,50), (0,50), (0, 50), (0, 1), (0, 1), (0, 1)]
    pbnds    = [(0, 2), (0, 2), (0, 3), (0, 10), (0, 1), (0, 1), (0, 1)]
    p_name   = ['α_STA', 'α_VOL', 'α_ACT', 'β', 'w0', 'w1', 'w2']
    n_params = len(bnds)
    voi      = ['p'] 

    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.alpha_act = params[2]
        self.beta      = params[3]
        self.w0        = params[4]
        self.w1        = params[5]
        self.w2        = params[6]

    def _policy(self):
        m0, m1 = self.buffer.sample('mag0','mag1')
        mag = np.array([m0, m1])
        u_A = self.w0*self.p_S*mag + self.w1*self.p_S + \
                self.w2*mag + (1-self.w0-self.w1-self.w2)*self.q_A 
        return softmax(self.beta*u_A)
