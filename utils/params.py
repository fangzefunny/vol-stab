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

    # CogSci
    if args.agent_name == 'RDModel':
        args.bnds        = ( ( 0, 1), ( 0, 1), ( 0, 20),)
        args.params_name = [ 'α_s', 'α_a', 'β',]
    elif args.agent_name == 'RDModel2':
        args.bnds        = ( ( 0, 1), ( 0, 1), ( 0, 20),
                             ( 0, 1), ( 0, 1), ( 0, 20),)
        args.params_name = [ 'α_s_stab', 'α_a_stab', 'β_stab',
                             'α_s_vol',  'α_a_vol',  'β_vol',]
    elif args.agent_name == 'SMModel':
        args.bnds        = ( ( 0, 1), ( 0, 20),
                             ( 0, 1), ( 0, 20),)
        args.params_name = [ 'α_s_stab', 'β_stab',
                             'α_s_vol',  'β_vol',]

    # basic 
    elif args.agent_name == 'NM':
        args.bnds        = ( ( 0, 1), 
                             ( 0, 1), 
                             ( 0, 5),)
        args.params_name = [ 'α_s_stab', 
                             'α_s_vol', 
                             'k',]
    elif args.agent_name == 'TM':
        args.bnds        = ( ( 0, 1), ( 1/20, 1000),  
                             ( 0, 1), ( 1/20, 1000),)
        args.params_name = [ 'α_s_stab', 'τ_stab', 
                             'α_s_vol',  'τ_vol', ]
    elif args.agent_name == 'SM':
        args.bnds        = ( ( 0, 1), ( 0, 1),  
                             ( 0, 1), ( 0, 1),
                             ( 0, 5), ( 1/20, 1000),)
        args.params_name = [ 'α_s_stab', 'α_τ_stab', 
                             'α_s_vol',  'α_τ_vol', 
                             'k',        'τ0',]
    
    ## With amortized inference
    elif args.agent_name == 'NMa':
        args.bnds        = ( ( 0, 1), ( 0, 1), 
                             ( 0, 1), ( 0, 1), 
                             ( 0, 5),)
        args.params_name = [ 'α_s_stab', 'α_a_stab',
                             'α_s_vol',  'α_a_vol',
                             'k',]
    elif args.agent_name == 'TMa':
        args.bnds        = ( ( 0, 1), ( 0, 1), ( 1/20, 1000), 
                             ( 0, 1), ( 0, 1), ( 1/20, 1000),)
        args.params_name = [ 'α_s_stab', 'α_a_stab', 'β_stab', 
                             'α_s_vol',  'α_a_vol',  'β_vol' ,]
    elif args.agent_name == 'SMa':
        args.bnds        = ( ( 0, 1), ( 0, 1), ( 0, 1),  
                             ( 0, 1), ( 0, 1), ( 0, 1), 
                             ( 0, 5), ( 1/20, 1000))
        args.params_name = [ 'α_s_stab', 'α_a_stab', 'α_τ_stab', 
                             'α_s_vol',  'α_a_vol',  'α_τ_vol', 
                             'k', 'τ0']
                             
    # if there is input initialization, we do not need to
    # random the intialization 
    if len(args.init) > 0:
        args.fit_num = 1 
    return args
  
