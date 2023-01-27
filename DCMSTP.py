import numpy as np

class DCMST():
    '''
    A class for representing and generating degree constrained minimum spanning tree problems
    '''
    def __init__(self, 
                 n, # number of vertices
                 a, b, # vertex degrees will be in the interval [a,b]
                 density, # graph density             
                 l, # vertex coordinatex will be in the interaval [0,l)
                 random=np.random.default_rng(0), # random number generator
                 ):
        self.n = n

        # generating edges
        edges = [(u, v) for u in range(n) for v in range(n) if random.random() < density]
        self.adj = np.zeros((n,n))
        for u, v in edges:
            self.adj[u,v] = self.adj[v,u] = 1

        # generating edge weights
        self.coords = random.random((n,2))*l
        self.weight = np.zeros((n, n))

        euc = lambda u, v: np.linalg.norm(self.coords[u]-self.coords[v])
        for u in range(n):
            for v in range(n):
                self.weight[u, v] = self.adj[u, v]*euc(u, v)

        # generating degrees
        self.max_degree = random.integers(low=a, high=b+1, size=n)

def Prim(inst, weight):
    n = inst.n

    added = np.zeros(n) # whether the vertex was added or not to the ST
    w = np.full(n, np.Inf) # minimum weight to reach each vertex
    p = np.zeros(n, dtype=int) # parents

    w[0], totalW, T = 0, 0, []

    for i in range(n):
        v = min(filter(lambda x: not added[x], range(n)), key = lambda x: w[x])
        added[v] = 1
        totalW += w[v]
        if v != 0: T.append((v, p[v]))

        for u in range(n):
            if inst.adj[u, v] and not added[u] and weight(v, u) < w[u]:
                w[u] = weight(v, u)
                p[u] = v

    return T, totalW

def LRP(instance, multipliers):
    b = multipliers['degree_constraints']
    mod_weight = lambda u, v: instance.weight[u, v] + b[u] + b[v]
    edges, obj = Prim(instance, mod_weight)
    obj -= np.matmul(b, instance.max_degree)
    return edges, obj

def get_subgradient(instance, sol):
    subg = np.zeros(instance.n)
    for u, v in sol:
        subg[u] += 1
        subg[v] += 1
    subg -= instance.max_degree 

    return subg

def example_1():
    import pylar.LR as LR
 
    rng = np.random.default_rng(0)

    inst = DCMST(n=100, a=1, b=3, density=1, l=1000, random=rng)
    edges, cost = Prim(inst, lambda u, v: inst.weight[u, v])


    ldp_sol, ldp_obj, ldp_mults = LR.subgradient(instance=inst,
                                                 lagrangian_subproblem_solver=LRP,
                                                 dualized_constraints={'degree_constraints':{'shape':(inst.n,), 'subgradient':get_subgradient, 'sense':'L'}},
                                                 stop_criterion=LR.stop.max_iter(1000),
                                                 stepsize=LR.step.constant_stepsize(0.1))


    print(ldp_obj)

def example_2():
    import pylar.LR as LR
 
    rng = np.random.default_rng(0)

    inst = DCMST(n=100, a=1, b=3, density=1, l=1000, random=rng)
    edges, cost = Prim(inst, lambda u, v: inst.weight[u, v])


    ldp_sol, ldp_obj, ldp_mults = LR.subgradient(instance=inst,
                                                 lagrangian_subproblem_solver=LRP,
                                                 dualized_constraints={'degree_constraints':{'shape':(inst.n,), 'subgradient':get_subgradient, 'sense':'L'}},
                                                 stop_criterion=LR.stop.max_iter(m=1000),
                                                 stepsize=LR.step.pipe([LR.step.square_summable_not_summable(a=1),
                                                                        LR.step.Polyak_estimated(default_estimate=20000)]),
                                                 direction=LR.direction.pipe([LR.direction.subgradient(),
                                                                              LR.direction.smooth(alpha=0.99)]))


    print(ldp_obj)

if __name__ == '__main__':
    #example_1()
    example_2()


