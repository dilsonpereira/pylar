import numpy as np
import matplotlib.pyplot as plt

def get_constant_stepsize(s=1.0):
    def f(*args, **kwargs): 
        return s

    return f

def get_constant_steplength(s=1.0):
    def f(direction, *args, **kwargs):
         norm = np.linalg.norm(direction)+1e-12
         return s/norm
    
    return f

def get_square_summable_not_summable(a=1.0, b=0.0):
    def f(k, *args, **kwargs):
         return a/(b+k)
    
    return f

def get_nonsummable_diminishing(a=1.0):
    def f(k, *args, **kwargs):
        return a/np.sqrt(k) 

    return f

def get_Polyak_steplength(opt_estimate, s=1.0):
    def f(obj, direction, *args, **kwargs): # original usa subgradient
        norm = np.linalg.norm(direction)+1e-12
        return s*(opt_estimate-obj)/(norm**2)

    return f

def get_max_iter_stop_criterion(max_iter=1000):
    def f(k, *args, **kwargs):
        return k > max_iter 

    return f

def get_direction_pipe(functions):
    def f(subgradient, *args, **kwargs):
        for g in functions:
            subgradient = g(subgradient, *args, **kwargs)

        return subgradient
    
    return f
    
def get_polyak_direction(alpha):
    prev = None 
    def f(subgradient, *args, **kwargs):
        nonlocal prev
        if not prev: prev = subgradient

        d = [(1-alpha)*p + alpha*s for p, s in zip(prev, subgradient)]

        return d

    return f

def get_normalized_subgradient():
    def f(subgradient, *args, **kwargs):
        norm = np.linalg.norm(subgradient)+1e-12
        return [s/norm for s in subgradient]

    return f

def get_subgradient():
    def f(subgradient, *args, **kwargs):
        return subgradient

    return f

def subgradient(instance, 
                lagrangian_subproblem_solver,
                dualized_constraints,
                stop_criterion=get_max_iter_stop_criterion(max_iter=1000),
                stepsize=get_constant_stepsize(1.0),
                direction=get_subgradient(),
                ):

    multipliers = [np.zeros(c['shape']) for c in dualized_constraints]
    senses = [c['sense'] for c in dualized_constraints]

    ldp_obj = -np.Inf
    hist = []

    k = 1
    while not stop_criterion(k):
        ls_sol, ls_obj = lagrangian_subproblem_solver(instance=instance, 
                                                      multipliers=multipliers)

        if ls_obj > ldp_obj:
            ldp_sol, ldp_obj, ldp_multipliers = ls_sol.copy(), ls_obj, [m.copy() for m in multipliers]
            #print(k, ldp_obj)

        #if k >= half and not LDP > hist[-half]:
        #    s /= 2

        hist.append(ldp_obj)

        subg = [c['subgradient'](instance=instance, 
                                 sol=ls_sol) for c in dualized_constraints]
        dir = direction(subg)

        s = stepsize(k=k, 
                     obj=ls_obj,
                     hist=hist, 
                     direction=dir)

        for mult, d, sense  in zip(multipliers, dir, senses):
            mult += s*d
            if sense == 'G':
                np.minimum(mult, 0, out=mult)
            elif sense == 'L':
                np.maximum(mult, 0, out=mult)
            
        k += 1

    plt.plot(hist)
    plt.show()
    print(ldp_obj)

    return ldp_sol, ldp_obj, ldp_multipliers
