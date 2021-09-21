import pandas as pd 
from scipy.stats import beta, uniform

from utils.brains import * 

# find the current path

def set_hyperparams(args):
    '''Set up hyperparams
    Based on the hypoerparameters from args parser,
    we the function choose the correct agent, parameters,
    and the corresponding bounds for parameter fitting. 
    '''
    args.init = []

    ## elife models
    args.brain = eval( args.brain_name)
    args.param_priors = None
    if args.brain_name == 'model1':
        args.bnds = ( ( .000, 1.), ( .1, 10.), ( .000, 20))
        args.params_name = [ 'α_s', 'γ', 'β']
    elif args.brain_name == 'model2':
        args.bnds = ( ( .000, 1.), ( .000, 1.), ( .000, 20))
        args.params_name = [ 'α_s', 'λ', 'β']
    elif args.brain_name == 'model7':
        args.bnds = ( ( .000, 1.), ( .000, 1.), ( .1, 10.), 
                      ( .000, 20))
        args.params_name = [ 'α_q', 'λ', 'r', 'β']
    elif args.brain_name == 'model8':
        args.bnds = ( ( .000, 1.), ( .000, 1.), ( .1, 10.), 
                      ( .000, 20.), ( .000, 1.))
        args.params_name = [ 'α_q', 'λ', 'r', 'β', 'ε']
    elif args.brain_name == 'model11':
        args.bnds = ( ( .000, 1.), ( .000,  1.), ( .1,  10.), 
                      ( .000, 1.), ( .000, 20.), ( .000, 20.))
        args.params_name = [ 'α_q', 'λ', 'r', 'α_a', 'β', 'β_a']
    
    ## model extensions
    elif args.brain_name == 'modelE1':
        args.bnds = ( ( .000, 1.), ( .000,  1.), ( .1,  10.), 
                      (  -1., 1.), ( .000, 20.), ( .000, 20.))
        args.params_name = [ 'α_q', 'λ', 'r', 'α_a', 'β', 'β_a']
    elif args.brain_name == 'modelE2':
        args.bnds = ( ( .000, 1.), ( .000,  1.), ( .1,  10.), 
                      ( .000, 1.), ( .000, 20.), ( .000, 20.), ( .000, 20.))
        args.params_name = [ 'α_q', 'λ', 'r', 'α_a', 'β', 'β_a', 'β_c']
    elif args.brain_name == 'RRmodel':
        args.bnds = ( ( .000, 1.), ( .000,  1.), ( 1e-4, 1.))
        args.params_name = [ 'α_q', 'α_a', 'τ']
        if args.fit_mode == 'map':
            args.param_priors = [ beta(3.5, 3), beta(3, 3.5), uniform(0, 20)]
    
    # if there is input initialization, we do not need to
    # random the intialization 
    if len(args.init) > 0:
        args.fit_num = 1 
    return args
  
