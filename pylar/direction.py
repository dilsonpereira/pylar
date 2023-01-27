import numpy as np

def pipe(functions):
    def f(direction, *args, **kwargs):
        for g in functions:
            direction = g(direction=direction, *args, **kwargs)

        return direction
    
    return f
    
def smooth(alpha=0.9):
    prev = None 
    def f(direction, *args, **kwargs):
        nonlocal prev
        if not prev: prev = direction

        d = {k:(1-alpha)*prev[k] + alpha*direction[k] for k in direction}

        prev = d

        return d

    return f

def norm(d):
    return np.sqrt( sum( np.sum(np.square(v)) for v in d.values() ) )+1e-12

def normalized():
    def f(direction, *args, **kwargs):
        n = norm(direction)
        return {k:d/n for k, d in direction.items()}

    return f

def identity():
    def f(direction, *args, **kwargs):
        return direction

    return f

subgradient = identity

