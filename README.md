# pylar: A python microframework for integer programming Lagrangian relaxation

The basic idea is to simplify Lagrangian relaxation implementation by requiring only a Lagrangian subproblem solver and functions to generate the subgradients.

Let's use the degree constrained minimum spanning tree problem as an example to illustrate how to use the framework.

## An example: the degree constrained minimum spanning tree problem
The degree constrained minimum spanning tree problem (DCMSTP) asks for the minimum spanning tree of a graph $G=(V,E)$, such that each vertex has a maximum number $b_v$ of allowed connections. It can be formulated as 
```math
\min \sum_{e \in E} w_ex_e
```
```math
x \in X
```
```math
\sum_{e \in \delta(v)} x_e \leq b_v, v \in V,
```
```math
x_e \in \{0,1\}, e \in E,
```

where $x_e$ is a binary decision variable corresponding to the inclusion of edge $e$ or not, and $w_e$ is an edge weight.

### A class for DCMSTPs
The following is a class for representing and generating random euclidean DCMSTPs:
```python
class DCMST():
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
```

### DCMSTP Lagrangian relaxation
The lagrangian relaxation of the degree constraints yields the following lagrangian relaxation problem (LRP):
```math
\begin{align}
LRP(\theta) & = \min \sum_{e \in E} w_ex_e + \sum_{v \in V} \theta_v(\sum_{e \in \delta(v)}x_e - b_v) \\
& = \min \sum_{e=\{u,v\} \in E} (w_e + \theta_u + \theta_v)x_e -\sum_{v \in V} \theta_vb_v,
\end{align}
```
```math
s.t. \quad x \in X.
```
where $\theta_v \geq 0$ is the lagrangian multiplier related to vertex $v$'s degree constraint.

The LRP consists in finding a minimum spanning tree of $G$ under modified edge costs $(w_e + \theta_u + \theta_v)$ and then subtracting $\sum_{v \in V} \theta_vb_v$ from its objective value. The following is the code we will pass to `pylar` to solve LRPs.
```python
def LRP(instance, multipliers):
    b = multipliers['degree_constraints']
    mod_weight = lambda u, v: instance.weight[u, v] + b[u] + b[v]
    edges, obj = Prim(instance, mod_weight)
    obj -= np.matmul(b, instance.max_degree)
    return edges, obj
```
`pylar` requires a function for solving LRPs, it will pass the problem instance and a dictionary of multipliers as arguments to it. The key of the dictionary is the name you specify for the relaxed constraint, and the value is a `numpy` array corresponding to the multipliers of that constraint. Your function should return the LRP solution (in any format, as `pylar` doesn't make direct use of it) followed by its (modified) objective value. 

We also need to pass a function for each relaxed constraint that, given a current LRP solution, returns the subgradient corresponding to that constraint. In the DCMSTP case, the subgradient corresponding to the current LRP solution and the degree constraints is $(\sum_{e \in \delta(v)}x_e - b_v)_{v \in V}$. The following is the function we will pass to `pylar` to obtain subgradients. `pylar` will pass the problem instance and the current LRP solution as arguments to our function. The function returns a `numpy` array:
```python
def get_subgradient(instance, sol):
    subg = np.zeros(instance.n)
    for u, v in sol:
        subg[u] += 1
        subg[v] += 1
    subg -= instance.max_degree 

    return subg
```

The following is a python implemenation of Prim's algorithm, used by the LRP function above:
```python
def Prim(inst, # DCMSTP instance 
         weight # edge weight function
         ):
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

    return T, totalW # edges in the tree and total weight
```

### Using pylar to solve the Lagrangian dual
Once we have the functions to solve Lagrangian subproblems and obtain subgradients, we are ready to call `pylar` to solve the Lagrangian dual
```math
\max_{\theta \geq 0} LRP(\theta):
```
```python
import pylar.LR as LR

ldp_sol, ldp_obj, ldp_mults = LR.subgradient(instance=inst,
                                             lagrangian_subproblem_solver=LRP,
                                             dualized_constraints={'degree_constraints':{'shape':(inst.n,), 'subgradient':get_subgradient, 'sense':'L'}},
                                             stop_criterion=LR.stop.max_iter(1000),
                                             stepsize=LR.step.constant_stepsize(0.1))
```
The problem instance `inst` could have been generated as follows:
```python
rng = np.random.default_rng(0)
inst = DCMST(n=100, a=1, b=3, density=1, l=1000, random=rng)
```
The code above invokes `pylar` to solve the lagrangian dual by the subgradient method, using a step size of 0.1 and 10000 iterations. The dualized constraints are passed as a dict argument. The keys of this dict are the names of your choice for the constraints. The value is another dict, containinig the shape of the subgradient, the function that returns its subgradient, and its sense ('G', 'L', or 'E').

To use a step size `s` in the direction of the normalized subgradient, the following could be passed as the `stepsize` parameter:
```python
stepsize=LR.step.pipe([LR.step.constant_stepsize(s),
                       LR.step.normalized_stepsize()])
```
or simply `stepsize=LR.step.contant_steplength(s)`.

If an upper bound was known, we could use a Polyak step length:
```python
stepsize=LR.step.pipe([LR.step.constant_stepsize(s),
                       LR.step.Polyak_steplength(ub)])
```
If we wanted to update the stepsize by a factor of $\alpha=0.1$ at every 100 iterations without improvement in the dual objective, starting with a step `s=1`, followed by a Polyak step length, we could do the following:
```python
stepsize=LR.step.pipe([LR.step.decreasing_tolerance(s=1, alpha=0.1, tol=100),
                       LR.step.Polyak_steplength(opt=ub)])
```
The subgradient function returns the solution attaining the best dual objective, its objective, and a dict of the corresponding lagrangian multipliers, indexed by the chosen names of their constraints.
