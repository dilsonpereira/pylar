import numpy as np

class DCMST():
    def __init__(self, n, a, b, l=1000, density=1, random=np.random):
        self.n = n

        self.adj = random.binomial(1, density, (n, n))
        for i in range(n):
            self.adj[i, i] = 0

        self.X = random.random(n)*l
        self.Y = random.random(n)*l
        self.weight = np.zeros((n, n))

        euc = lambda u, v: ((self.X[u]-self.X[v])**2+(self.Y[u]-self.Y[v])**2)**0.5
        for u in range(n):
            for v in range(n):
                self.weight[u, v] = self.adj[u, v]*euc(u, v)

        V = [v for v in range(n)]
        random.shuffle(V)
        degree_sum = 2*n-2
        self.maxDegree = np.zeros(n)
        i = n
        for v in V:
            i -= 1
            if i == 0:
                self.maxDegree[v] = degree_sum
            else:
                max_degree = degree_sum - i
                self.maxDegree[v] = random.integers(min(a, max_degree), min(b, max_degree)+1)
            degree_sum -= self.maxDegree[v]

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
    multipliers = multipliers[0]
    lagWeights = lambda u, v: instance.weight[u, v] + multipliers[u] + multipliers[v]
    edges, lrp = Prim(instance, lagWeights)
    lrp -= np.matmul(multipliers, instance.maxDegree)
    return edges, lrp

def get_subgradient(instance, sol):
    subg = np.zeros(instance.n)
    for u, v in sol:
        subg[u] += 1
        subg[v] += 1
    subg -= instance.maxDegree 

    return subg

if __name__ == '__main__':
    import LR

    rng = np.random.default_rng(0)

    inst = DCMST(n=50, a=1, b=4, random=rng)
    edges, cost = Prim(inst, lambda u, v: inst.weight[u, v])


    ldp_sol, ldp_obj, ldp_mults = LR.subgradient(instance=inst,
                                                 lagrangian_subproblem_solver=LRP,
                                                 dualized_constraints=[{'shape':(inst.n,), 'subgradient':get_subgradient, 'sense':'E'}],
                                                 stepsize=LR.get_Polyak_steplength(opt_estimate=8000),
                                                 direction=LR.get_direction_pipe([LR.get_subgradient(),
                                                                                  LR.get_polyak_direction(alpha=0.9)]))


