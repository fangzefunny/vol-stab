import numpy as np
import torch
from torch.distributions import Beta
from scipy.special import softmax
from scipy.stats import norm, gamma, beta

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
    p_priors = None 
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
    voi      = ['ps', 'pi'] 
   
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
        alpha = self.alpha_sta if c=='sta' else self.alpha_vol
        self.p += alpha * (o - self.p)
        self.p_S = np.array([1-self.p, self.p])

    def _policy(self):
        m1, m2 = self.buffer.sample('mag0','mag1')
        mag = np.array([m1, m2])
        return softmax(self.beta * self.p_S * mag)

    def print_ps(self):
        return self.p

    def print_pi(self):
        return self._policy()[1]

class gagModel(gagRL):
    name     = 'Gagne best model'
    bnds     = [(0, 1), (0, 1), (0, 50), (0, 50), 
                (0, 1), (0, 50), (0, 1), (0, 1), (0, 1)]
    pbnds    = [(0,.5), (0,.5), (0, 10), (0, 10), 
                (0, 1), (0, 10), (0, 1), (0, 1), (0, 1)]
    p_name   = ['α_STA', 'α_VOL', 'β_STA', 'β_VOL', 
                'α_ACT', 'β_ACT', 'λ_STA', 'λ_ACT', 'r']  
    p_priors = [beta(a=2, b=2), beta(a=2, b=2), gamma(a=3, scale=3), gamma(a=3, scale=3),
                beta(a=2, b=2), gamma(a=3, scale=3), beta(a=2, b=2), beta(a=2, b=2), beta(a=2, b=2)]
    n_params = len(bnds)
    voi      = ['ps', 'pi'] 
   
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.beta_sta  = params[2]
        self.beta_vol  = params[3]
        self.alpha_act = params[4]
        self.beta_act  = params[5]
        self.lamb_sta  = params[6]
        self.lamb_vol  = params[7]
        self.r         = params[8]
    
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
        lamb = eval(f'self.lamb_{c}')
        v    = lamb*(self.p - (1-self.p)) \
               + (1-lamb)*abs(m1-m0)**self.r*np.sign(m1-m0)
        va   = eval(f'self.beta_{c}')*v + self.beta_act*(self.q - (1-self.q))
        pa   = 1 / (1 + np.exp(-va))
        return np.array([1-pa, pa])

class risk(gagRL):
    name     = 'Gagne RL'
    bnds     = [(0, 1), (0, 1), (0, 30), (0, 20), (0, 20)]
    pbnds    = [(0,.5), (0,.5), (0, 10), (0, 20), (0, 20)]
    p_name   = ['α_STA', 'α_VOL', 'β', 'γ_STA', 'γ_VOL']  
    n_params = len(bnds)
    p_priors = [beta(a=2, b=2), beta(a=2, b=2), gamma(a=3, scale=3), gamma(a=3, scale=3), gamma(a=3, scale=3)]
    voi      = ['ps', 'pi'] 
   
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.beta      = params[2]
        self.gamma_sta = params[3]
        self.gamma_vol = params[4]
    
    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt','state')
        self.p += eval(f'self.alpha_{c}') * (o - self.p)
        ps = np.clip(eval(f'self.gamma_{c}')*(self.p-.5)+.5, 0, 1)
        self.p_S = np.array([1-ps, ps])

# ---------  min CE ----------- #

sigmoid = lambda x: 1 / (1+np.exp(-x))

class ceModel(gagRL):
    name     = 'ce RL'
    bnds     = [(0,30), (0,30), (0, 30)]
    pbnds    = [(0, 2), (0, 2), (0, 10)]
    p_name   = ['α_STA', 'α_VOL', 'β']
    n_params = len(bnds)
    voi      = ['ps', 'pi'] 

    def _init_Critic(self):
        self.theta = 0 
        self.p     = sigmoid(self.theta)
        self.p_S   = np.array([1-self.p, self.p]) 

    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt','state')
        alpha = self.alpha_sta if c=='sta' else self.alpha_vol
        self.theta += alpha * (o - self.p)
        self.p = sigmoid(self.theta)
        self.p_S = np.array([1-self.p, self.p])

    def print_ps(self):
        return self.p

    def print_pi(self):
        return self._policy()[1]

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
        lamb = self.lamb_sta if c=='sta' else self.lamb_vol
        mag = np.array([m0, m1])
        u_A = self.p_S*mag 
        return softmax(self.beta*u_A + lamb*np.log(self.q_A+eps_))

class mix(CSCE):
    name     = 'ce RL'
    bnds     = [(0,50), (0,50), (0,50), (0, 50), (0, 1), (0, 1), (0, 1)]
    pbnds    = [(0, 2), (0, 2), (0, 3), (0, 10), (0, 1), (0, 1), (0, 1)]
    p_name   = ['α_STA', 'α_VOL', 'α_ACT', 'β', 'w0', 'w1', 'w2']
    n_params = len(bnds)
    voi      = ['ps', 'pi'] 

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

class mix_Explore(mix):
    name     = 'mix, fit different'
    bnds     = [(0,50), (0,50), (0,50), (0,50), 
                (0, 1), (0, 1), (0, 1), (0, 1),
                (0, 1), (0, 1), (0, 1), (0, 1)]
    pbnds    = [(0, 2), (0, 2), (0, 3), (0, 5),
                (0, 1), (0, 1), (0, 1), (0, 1),
                (0, 1), (0, 1), (0, 1), (0, 1)]
    p_name   = ['α_STA', 'α_VOL', 'α_ACT', 'β',
                'w0_STA', 'w1_STA', 'w2_STA', 'w3_STA',
                'w0_VOL', 'w1_VOL', 'w2_VOL', 'w3_VOL']
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'alpha', 'w0', 'w1', 'w2', 'w3'] 
    
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.alpha_act = params[2]
        self.beta      = params[3]
        self.w0_sta    = params[4]
        self.w1_sta    = params[5]
        self.w2_sta    = params[6]
        self.w3_sta    = params[7]
        self.w0_vol    = params[8]
        self.w1_vol    = params[9]
        self.w2_vol    = params[10]
        self.w3_vol    = params[11]

    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        mag = np.array([m0, m1])
        u_A = eval(f'self.w0_{c}')*self.p_S*mag\
              + eval(f'self.w1_{c}')*self.p_S\
              + eval(f'self.w2_{c}')*mag\
              + eval(f'self.w3_{c}')*self.q_A
        return softmax(self.beta*u_A)

    def print_w0(self):
        return eval(f'self.w0_{self.buffer.sample("ctxt")}') 

    def print_w1(self):
        return eval(f'self.w1_{self.buffer.sample("ctxt")}')  

    def print_w2(self):
        return eval(f'self.w3_{self.buffer.sample("ctxt")}')  

    def print_w3(self):
        return eval(f'self.w3_{self.buffer.sample("ctxt")}')  
    
    def print_alpha(self):
        return eval(f'self.alpha_{self.buffer.sample("ctxt")}') 

class mix_red(mix_Explore):
    name     = 'mix, reduce beta'
    bnds     = [(0,50), (0,50), (0,50), 
                (0,40), (0,40), (0,40), (0,40),
                (0,40), (0,40), (0,40), (0,40)]
    pbnds    = [(0, 2), (0, 2), (0, 3), 
                (0, 2), (0, 2), (0, 2), (0, 2),
                (0, 2), (0, 2), (0, 2), (0, 2)]
    p_name   = ['α_STA', 'α_VOL', 'α_ACT',
                'w0_STA', 'w1_STA', 'w2_STA', 'w3_STA',
                'w0_VOL', 'w1_VOL', 'w2_VOL', 'w3_VOL']
    n_params = len(bnds)

    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.alpha_act = params[2]
        self.w0_sta    = params[3]
        self.w1_sta    = params[4]
        self.w2_sta    = params[5]
        self.w3_sta    = params[6]
        self.w0_vol    = params[7]
        self.w1_vol    = params[8]
        self.w2_vol    = params[9]
        self.w3_vol    = params[10]

    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        mag = np.array([m0, m1])
        u_A = eval(f'self.w0_{c}')*self.p_S*mag\
              + eval(f'self.w1_{c}')*self.p_S\
              + eval(f'self.w2_{c}')*mag\
              + eval(f'self.w3_{c}')*self.q_A 
        return softmax(u_A)

class mix_pol(mix_Explore):
    name     = 'mix, reduce beta'
    bnds     = [(0,50), (0,50), (0,50), (0,50),
                (-40,40), (-40,40), (-40,40), (-40,40),
                (-40,40), (-40,40), (-40,40), (-40,40)]
    pbnds    = [(0, 2), (0, 2), (0, 3), (0, 5),
                (-5, 5), (-5, 5), (-5, 5), (-5, 5),
                (-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    p_name   = ['α_STA', 'α_VOL', 'α_ACT', 'β',
                'λ0_STA', 'λ1_STA', 'λ2_STA', 'λ3_STA',
                'λ0_VOL', 'λ1_VOL', 'λ2_VOL', 'λ3_VOL']
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'alpha', 'w1', 'w2', 'w3', 'w4', 'l1', 'l2', 'l3', 'l4']

    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.alpha_act = params[2]
        self.beta      = params[3]
        self.l0_sta    = params[4]
        self.l1_sta    = params[5]
        self.l2_sta    = params[6]
        self.l3_sta    = params[7]
        self.l0_vol    = params[8]
        self.l1_vol    = params[9]
        self.l2_vol    = params[10]
        self.l3_vol    = params[11]

    def get_w(self, c):
        l0 = eval(f'self.l0_{c}')
        l1 = eval(f'self.l1_{c}')
        l2 = eval(f'self.l2_{c}')
        l3 = eval(f'self.l3_{c}')
        return softmax([l0, l1, l2, l3])

    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        mag = np.array([m0, m1])
        p_SM = softmax(self.beta*self.p_S*mag)
        p_M  = softmax(self.beta*mag)
        w0, w1, w2, w3 = self.get_w(c)
        # creat the mixature model 
        return w0*p_SM + w1*self.p_S + w2*p_M + w3*self.q_A 

    def print_w1(self):
        return self.get_w(self.buffer.sample("ctxt"))[0]

    def print_w2(self):
        return self.get_w(self.buffer.sample("ctxt"))[1] 

    def print_w3(self):
        return self.get_w(self.buffer.sample("ctxt"))[2]  

    def print_w4(self):
        return self.get_w(self.buffer.sample("ctxt"))[3]  

    def print_l1(self):
        return eval(f'self.l0_{self.buffer.sample("ctxt")}')

    def print_l2(self):
        return eval(f'self.l1_{self.buffer.sample("ctxt")}')

    def print_l3(self):
        return eval(f'self.l2_{self.buffer.sample("ctxt")}')

    def print_l4(self):
        return eval(f'self.l3_{self.buffer.sample("ctxt")}')

class mix_pol_3w(mix_pol):
    name     = 'mix, policy reduce'
    bnds     = [(0,50), (0,50), (0,50), (0,50),
                (-40,40), (-40,40), (-40,40),
                (-40,40), (-40,40), (-40,40)]
    pbnds    = [(0, 2), (0, 2), (0, 3), (0, 5),
                (-5, 5), (-5, 5), (-5, 5),
                (-5, 5), (-5, 5), (-5, 5),]
    p_name   = ['α_STA', 'α_VOL', 'α_ACT', 'β',
                'λ0_STA', 'λ1_STA', 'λ2_STA',
                'λ0_VOL', 'λ1_VOL', 'λ2_VOL']
    p_priors = [gamma(a=3, scale=3), gamma(a=3, scale=3),gamma(a=3, scale=3),gamma(a=3, scale=3),
                norm(loc=0, scale=15), norm(loc=0, scale=15), norm(loc=0, scale=15), norm(loc=0, scale=15),
                norm(loc=0, scale=15), norm(loc=0, scale=15), norm(loc=0, scale=15), norm(loc=0, scale=15)]
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'alpha', 'w1', 'w2', 'w3', 'l1', 'l2', 'l3']

    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.alpha_act = params[2]
        self.beta      = params[3]
        self.l0_sta    = params[4]
        self.l1_sta    = params[5]
        self.l2_sta    = params[6]
        self.l0_vol    = params[7]
        self.l1_vol    = params[8]
        self.l2_vol    = params[9]

    def get_w(self, c):
        l0 = eval(f'self.l0_{c}')
        l1 = eval(f'self.l1_{c}')
        l2 = eval(f'self.l2_{c}')
        return softmax([l0, l1, l2])

    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        mag = np.array([m0, m1])
        p_SM = softmax(self.beta*self.p_S*mag)
        p_M  = softmax(self.beta*mag)
        w0, w1, w2 = self.get_w(c)
        # creat the mixature model 
        return w0*p_SM + w1*p_M + w2*self.q_A 

class mixNN(mix):
    name     = 'mix, NN implementation'
    bnds     = [(0,50), (0,50), (0,50), (0, 50), (0, 1), (0, 1), (0, 1)]
    pbnds    = [(0, 2), (0, 2), (0, 3), (0, 10), (0, 1), (0, 1), (0, 1)]
    p_name   = ['α_STA', 'α_VOL', 'α_ACT', 'β', 'w0', 'w1', 'w2']
    n_params = len(bnds)
    voi      = ['ps', 'pi'] 

    #  ------- init ------ #

    def _init_Critic(self):
        self.mu  = (torch.ones([2,])*0.).requires_grad_()

    def _init_Actor(self):
        self.muA = (torch.ones([2,])*0.).requires_grad_()
    
    #  ------- forward ------ #

    def _policy(self):
        m1, m2 = self.buffer.sample('mag0','mag1')
        M = torch.tensor([m1, m2])
        # rewarding probability
        self.theta = self.p_t()
        # perseveration 
        self.phi   = self.pi0_A()
        # response policy 
        pi_A1M = self.pi_A1Mt(M, self.theta.detach()) 

        self.pi_A1M = pi_A1M.numpy()

        return self.pi_A1M

    def p_t(self):
        '''p(θ)'''
        return torch.softmax(self.mu, dim=0)

    def pi0_A(self):
        return torch.softmax(self.muA, dim=0)

    def pi_A1Mt(self, m, theta):
        '''π(a|m, θ)'''
        logit = self.w0*theta*m + self.w1*theta + self.w2*m \
                + (1-self.w0-self.w1-self.w2)*self.phi.detach() 
        return torch.softmax(self.beta*logit, dim=0)

    #  ------- backward ------ #

    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt', 'state')
        alpha = self.alpha_sta if c=='sta' else self.alpha_vol
        # calculate loss 
        thetaTar = torch.eye(self.nA)[o, :]
        loss = -(thetaTar * (self.theta+eps_).log()).sum()
        loss.backward()
        # step 
        self.mu.data -= alpha * self.mu.grad.data
        self.mu.grad.data.zero_()

    def _learnActor(self):
        a = self.buffer.sample('act')
        phiTar = torch.eye(self.nA)[a, :]
        loss = -(phiTar * (self.phi+eps_).log()).sum()
        loss.backward()
        self.muA.data -= self.alpha_act * self.muA.grad.data
        self.muA.grad.data.zero_()
     
    #  ------- visualization ------ #

    def print_ps(self):
        return self.theta.detach().numpy()[1]

    def print_pi(self):
        return self.pi_A1M[1]

class mixNN2(mixNN):

    #  ------- init ------ #
    def _init_Critic(self):
        self.mu  = (torch.ones([1,])*0.).requires_grad_()

    def _init_Actor(self):
        self.muA = (torch.ones([1,])*0.).requires_grad_()

    #  ------- forward ------- #
    def p_t(self):
        '''p(θ)'''
        return torch.sigmoid(self.mu)

    def pi0_A(self):
        return torch.sigmoid(self.muA)

    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt', 'state')
        alpha = self.alpha_sta if c=='sta' else self.alpha_vol
        # calculate loss 
        thetaTar = torch.tensor(o)
        loss = -(thetaTar * (self.theta+eps_).log() +
                 (1-thetaTar) * (1-self.theta+eps_).log())
        loss.backward()
        # step 
        self.mu.data -= alpha * self.mu.grad.data
        self.mu.grad.data.zero_()

    def _learnActor(self):
        a = self.buffer.sample('act')
        phiTar = torch.tensor(a)
        loss = -(phiTar * (self.phi+eps_).log() +
                 (1-phiTar) * (1-self.phi+eps_).log())
        loss.backward()
        self.muA.data -= self.alpha_act * self.muA.grad.data
        self.muA.grad.data.zero_()

    def pi_A1Mt(self, m, theta):
        '''π(a|m, θ)'''
        p_S = torch.tensor([1-theta, theta])
        q_A = torch.tensor([1-self.phi.detach(), self.phi.detach()])
        logit = self.w0*p_S*m + self.w1*p_S + self.w2*m \
                + (1-self.w0-self.w1-self.w2)*q_A
        return torch.softmax(self.beta*logit, dim=0)

    def print_ps(self):
        return self.theta.detach().numpy()

    def print_pi(self):
        return self.pi_A1M[1]

# --------- epstemic uncertainty ----------- #

class distRL(baseAgent):
    name     = 'Distributional RL'
    bnds     = [(0, 50), (0,20), (0, 10)]
    pbnds    = [(0, 1), (0, 5), (0, 2)]
    p_name   = ['α', 'β', 'logv']  
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'vars']  

    def load_params(self, params):
        self.alpha = params[0]
        self.beta  = params[1]
        self.logv  = params[2]
        self.N     = 5000

    #  ------- init ------ #

    def _init_Critic(self):
        self.loga = (torch.ones([1,])*self.logv).requires_grad_()
        self.logb = (torch.ones([1,])*self.logv).requires_grad_()

    #  ------- forward ------ #

    def _policy(self):
        m1, m2 = self.buffer.sample('mag0','mag1')
        M = torch.tensor([m1, m2])
        # predict the rewarding probability
        self.theta = self.p_t()
        # response policy 
        pi_A1M = self.pi_A1Mt(M, self.theta.detach()).mean(0) 

        return pi_A1M.numpy()

    def p_t(self):
        '''p(θ)'''
        self.a = torch.clamp(self.loga.exp(), .01, 50)
        self.b = torch.clamp(self.logb.exp(), .01, 50)

        return Beta(self.a, self.b).rsample([self.N,])

    def pi_A1Mt(self, m, theta):
        '''π(a|m, θ)'''
        logit = theta.detach()*m.unsqueeze(0)
        return torch.softmax(self.beta*logit, dim=1)

    #  ------- backward ------ #

    def learn(self):
        self._learnCritic()

    def _learnCritic(self):
        o = self.buffer.sample('state')
        # calculate loss 
        # get the log variational posterior 
        log_post = Beta(self.a.detach(), self.b.detach()
                    ).log_prob(self.theta).mean()
        # get the log like: olog(θ) + (1-o)log(1-θ)

        log_like = (o*(self.theta).log() + 
                    (1-o)*(1-self.theta).log()).mean()
        loss = log_post - log_like
        loss.backward()
        # step 
        self.loga.data -= self.alpha * self.loga.grad.data
        self.logb.data -= self.alpha * self.logb.grad.data
        self.loga.grad.data.zero_()
        self.logb.grad.data.zero_()

    #  ------- visualization ------ #

    def print_ps(self):
        return self.theta.detach().numpy().mean(0)

    def print_pi(self):
        return self._policy()[1]

    def print_vars(self):
        return self.theta.detach().numpy().var()

class distRL_Mix(distRL):
    name     = 'Distributional RL, mix'
    bnds     = [(0, 2), (0, 2), (0, 5), (0, 5), (0, 2), (0, 1), (0, 1), (0, 1)]
    pbnds    = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 1), (0, 1), (0, 1), (0, 1)]
    p_name   = ['α_MU', 'α_SIG', 'β', 'logσ0', 'α_ACT', 'w0', 'w1', 'w2']  
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'vars']  

    def load_params(self, params):
        self.alpha_mu  = params[0]
        self.alpha_sig = params[1]
        self.beta      = params[2]
        self.logsig0   = params[3]
        self.alpha_act = params[4]
        self.w0        = params[5]
        self.w1        = params[6]
        self.w2        = params[7]
        self.N       = 100

    #  ------- init -------  #

    def _init_Actor(self):
        self.muA = (torch.ones([2,])*0.).requires_grad_()

    #  ------- forward ------ #

    def _policy(self):
        m1, m2 = self.buffer.sample('mag0','mag1')
        m = torch.tensor([m1, m2])
        # predict the rewarding probability
        self.theta = self.p_t()
        # predict the perseveration
        self.phi   = self.pi0_A()
        # response policy 
        pi_A1m = self.pi_A1Mt(m, self.theta.detach()).mean(0) 

        return pi_A1m.numpy()

    def pi0_A(self):
        return torch.softmax(self.muA, dim=0)

    def pi_A1Mt(self, m, theta):
        '''π(a|m, θ)'''
        logit = self.w0*theta*m.unsqueeze(0) \
                + self.w1*theta \
                + self.w2*m.unsqueeze(0) \
                + (1-self.w0-self.w1-self.w2)*self.phi.detach().unsqueeze(0)
        return torch.softmax(self.beta*logit, dim=1)

    #  ------- backward ------ #

    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _learnActor(self):
        a = self.buffer.sample('act')
        phiTar = torch.eye(self.nA)[a, :]
        loss = - (phiTar * (self.phi+eps_).log()).sum()
        loss.backward()
        self.muA.data -= self.alpha_act * self.muA.grad.data
        self.muA.grad.data.zero_()

        