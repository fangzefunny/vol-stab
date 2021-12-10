import pandas as pd 
from scipy.stats import beta, uniform

from utils.agents import * 

# find the current path

def set_hyperparams(args):
    '''Set up hyperparams
    Based on the hypoerparameters from args parser,
    we the function choose the correct agent, parameters,
    and the corresponding bounds for parameter fitting. 

    Note: It is very dangerous to use (.0, 1.), becasue the
    computer may use 1.000001 using the float type, use
    (0, 1) all the time. 
    '''
    args.init = []

    ## elife models
    args.agent = eval( args.agent_name)
    args.param_priors = None
    if args.agent_name == 'model1':
        args.bnds = ( ( 0, 1), ( 0, 1), ( .1, 10.), ( 0, 20))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'γ', 'β']
    elif args.agent_name == 'model2':
        args.bnds = ( ( 0, 1), ( 0, 1), ( 0, 1), ( 0, 20))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'λ', 'β']
    elif args.agent_name == 'model7':
        args.bnds = ( ( 0, 1), ( 0, 1), ( 0, 1), ( .1, 10.), 
                      ( 0, 20))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'λ', 'r', 'β']
    elif args.agent_name == 'model8':
        args.bnds = ( ( 0, 1),  ( 0, 1), ( 0, 1), ( .1, 10.), 
                      ( 0, 20.), ( 0, 1))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'λ', 'r', 'β', 'ε']
    elif args.agent_name == 'model11':
        args.bnds = ( ( 0, 1), ( 0,  1), ( 0,  1), ( .1,  10.), 
                      ( 0, 1), ( 0, 20.), ( 0, 20.))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'λ', 'r', 'α_a', 'β', 'β_a']
    
    ## model extensions
    elif args.agent_name == 'model11_m':
        args.bnds = ( ( 0,  1), ( .1,  10.), 
                      ( 0, 1), ( 0, 20.), ( 0, 20.))
        args.params_name = [ 'λ', 'r', 'α_a', 'β', 'β_a']
    elif args.agent_name == 'max_mag':
        args.bnds = [( 0,  20.)]
        args.params_name = [ 'β',]
    elif args.agent_name == 'RRmodel':
        args.bnds = ( ( 0, 1), ( 0, 1), ( 0,  1), ( .00, 20.))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'α_a', 'β']
        if args.fit_mode == 'map':
            args.param_priors = [ beta(3.5, 3), beta(3, 3.5), uniform(0, 20)]
    elif args.agent_name == 'RRmodel_f1':
        args.bnds = ( ( 0, 1), ( 0, 1), ( 0,  1), ( .00, 20.), ( .00, 10.))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'α_a', 'β', 'γ']
    elif args.agent_name == 'RRmodel_f2':
        args.bnds = ( ( 0, 1), ( 0, 1), ( 0,  1), ( .00, 20.), ( .00, 1))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'α_a', 'β', 'γ']
    elif args.agent_name == 'dual_sys':
        args.bnds = ( ( 0, 1), ( 0, 1), ( 0,  1), ( .00, 1), 
                      ( .00, 20.), ( .00, 10.))
        args.params_name = [ 'α_s_stab', 'α_s_vol', 'w', 'α_a', 'β', 'γ']


    # if there is input initialization, we do not need to
    # random the intialization 
    if len(args.init) > 0:
        args.fit_num = 1 
    return args
  
