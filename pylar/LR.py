import numpy as np
import pylar.stop as stop
import pylar.direction as direction
import pylar.step as step
import pylar.update as update
from copy import copy

def subgradient(instance, 
                lagrangian_subproblem_solver,
                dualized_constraints,
                stop_criterion=stop.max_iter(m=1000),
                stepsize=step.constant_stepsize(1.0),
                direction=direction.subgradient(),
                primal_heuristic=None,
                maximization=False,
                mult_warm_start=None
                ):

    dualized_constraints = {k:copy(c) for k, c in dualized_constraints.items()}
    for c in dualized_constraints.values():
        if 'update' not in c:
            c['update'] = update.simple()

    if mult_warm_start == None:
        multipliers = {k:np.zeros(c['shape']) for k, c in dualized_constraints.items()}
    else:
        multipliers = {k:np.copy(m) for k, m in mult_warm_start.items()}

    ldp_obj = np.Inf if maximization else -np.Inf
    hist = []

    primal_obj = -np.Inf if maximization else np.Inf
    p_sol = None

    it = 1
    while not stop_criterion(k=it, dual_obj=ldp_obj):
        ls_sol, ls_obj = lagrangian_subproblem_solver(instance=instance, 
                                                      multipliers=multipliers)

        if primal_heuristic != None:
            p_sol, p_obj = primal_heuristic(instance=instance,
                                            lagrangian_subproblem_solution=ls_sol,
                                            lagrangian_multipliers=multipliers)
            if (p_obj < primal_obj and not maximization) or (p_obj > primal_obj and maximization):
                primal_sol, primal_obj = p_sol, p_obj

        if (ls_obj > ldp_obj and not maximization) or (ls_obj < ldp_obj and maximization):
            ldp_sol, ldp_obj, ldp_multipliers = copy(ls_sol), ls_obj, {k:np.copy(m) for k, m in multipliers.items()}

        hist.append(ldp_obj)

        sign = -1 if maximization else 1
        subg = {k:sign * c['subgradient'](instance=instance, 
                                             sol=ls_sol) for k, c in dualized_constraints.items()}

        dir = direction(direction=subg, 
                        maximization=maximization)

        s = stepsize(k=it, 
                     dual_obj=ldp_obj,
                     hist=hist, 
                     direction=dir,
                     primal_obj=primal_obj,
                     maximization=maximization)

        multipliers = {k:c['update'](multipliers=multipliers[k],
                                     s=s,
                                     direction=dir[k],
                                     sense=c['sense'],
                                     maximization=maximization) for k, c in dualized_constraints.items()}

        it += 1

    return ldp_sol, ldp_obj, ldp_multipliers
