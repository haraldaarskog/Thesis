import numpy as np
from ypstruct import structure
import sys

def run(problem, params):

    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    nc = int(np.round(npop/2)*2)
    mu = params.mu
    sigma = params.sigma

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.values = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = 0 #np.inf
    bestsol.values = None

    # Initialize Population
    # returns array of structures. Array of npop individuals
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].values = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].values)
        if pop[i].cost > bestsol.cost:
            bestsol = pop[i].deepcopy()
            bestsol.values = pop[i].values
            print("Initialization: Best score = {}".format(bestsol.cost))
    # Best Cost of Iterations
    bestcost = np.empty(maxit)

    print("Initialization complete")

    # Main Loop
    for it in range(maxit):

        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        sum_cost = np.sum(costs)
        #if avg_cost != 0:
        #    costs = costs/avg_cost
        if sum_cost != 0:
            costs = costs/sum_cost
        probs = costs#np.exp(-beta*costs)

        popc = []
        for _ in range(nc//2):

            # Select Parents
            #q = np.random.permutation(npop)
            #p1 = pop[q[0]]
            #p2 = pop[q[1]]


            # Perform Roulette Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            # Perform Crossover
            c1, c2 = crossover(p1, p2)

            # Perform Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Apply Bounds
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)

            # Evaluate First Offspring
            c1.cost = costfunc(c1.values)
            if c1.cost > bestsol.cost:
                bestsol = c1.deepcopy()
                bestsol.values = c1.values

            # Evaluate Second Offspring
            c2.cost = costfunc(c2.values)
            if c2.cost > bestsol.cost:
                bestsol = c2.deepcopy()
                bestsol.values = c2.values

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)


        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost, reverse=True)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print("Iteration {}: Best score = {}".format(it, bestcost[it]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    #print(bestsol.values)
    return out

def crossover(p1, p2):
    sh = p1.values.shape
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.uniform(0, 1, sh)
    #Høyere alpha gir mer vekt på p1 sine verdier. Nærmere 0.5, 50% av hver. mest crossover
    c1.values = alpha*p1.values + (1-alpha)*p2.values
    c2.values = alpha*p2.values + (1-alpha)*p1.values
    return c1, c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    y.values = x.values
    flag = np.random.rand(*x.values.shape) <= mu
    ind = np.argwhere(flag)
    i_shape = ind.shape
    for i in ind:
        y.values[tuple(i)] += sigma*np.random.rand()
    return y

def apply_bound(x, varmin, varmax):
    x.values = np.maximum(x.values, varmin)
    x.values = np.minimum(x.values, varmax)

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


"""
a = arr.shape
print(len(a))
print(a, *a)
print(np.random.randn(*a))
arr = np.array([
[1.,2.,3.],
[4.,5.,6.]])
"""
