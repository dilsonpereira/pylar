import numpy as np

def simple():
    def f(multipliers, s, direction, sense, maximization, *args, **kwargs):
        multipliers += s*direction
        if (sense == 'G' and not maximization) or (sense=='L' and maximization):
            np.minimum(multipliers, 0, out=multipliers)
        elif (sense == 'L' and not maximization) or (sense=='G' and maximization):
            np.maximum(multipliers, 0, out=multipliers)

        return multipliers

    return f


