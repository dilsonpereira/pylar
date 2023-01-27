import numpy as np
from pylar.direction import norm

def pipe(functions):
    #Return value of each function will be passed to the next as parameter s
    def f(*args, **kwargs):
        s = functions[0](*args, **kwargs)
        for g in functions[1:]:
            s = g(s=s, *args, **kwargs)

        return s

    return f

def constant_stepsize(s=1.0):
    def f(*args, **kwargs): 
        return s

    return f

def decreasing_tolerance(s=1.0, alpha=0.5, tol=100):
    hist = []
    k = 0
    def f(dual_obj, maximization, *args, **kwargs): 
        nonlocal s
        nonlocal k

        if (k >= tol) and ((dual_obj < hist[-k]+1e-6 and not maximization) or (dual_obj > hist[-k]-1e-6 and maximization)):
            s *= alpha
            k = 0

        hist.append(dual_obj)
        k += 1

        return s

    return f

def normalized_stepsize():
    def f(direction, s=1.0, *args, **kwargs):
         n = norm(direction)
         return s/n
    
    return f

def constant_steplength(s=1.0):
    return pipe([constant_stepsize(s=s), 
                 normalized_stepsize()])

def square_summable_not_summable(a=1.0, b=0.0):
    def f(k, s=1.0, *args, **kwargs):
        return s*a/(b+k)
    
    return f

def nonsummable_diminishing(a=1.0):
    def f(k, s=1.0, *args, **kwargs):
        return s*a/np.sqrt(k) 

    return f

def Polyak_steplength(opt):
    def f(dual_obj, direction, s=1.0, *args, **kwargs):
        n = norm(direction)
        return s*(abs(opt()-dual_obj))/(n**2)

    return f

def Polyak_estimated(default_estimate, gamma=constant_stepsize(s=0.0)):
    def f(dual_obj, direction, primal_obj, s=1.0, *args, **kwargs):
        if primal_obj == None or primal_obj == np.inf:
            primal_obj = default_estimate
        n = norm(direction)
        g = gamma(dual_obj=dual_obj, direction=direction, primal_obj=primal_obj, s=s, *args, **kwargs)
        return s*(abs(primal_obj-dual_obj)+g)/(n**2)

    return f


