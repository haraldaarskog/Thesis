import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import model_functions as mf
import main_model as mm

# Sphere Test Function
def sphere(x):
    return sum(x**2)


M = 10
weeks = 2
diagnostic_processes = 2

number_of_queues = mf.get_total_number_of_queues()

def optimize_ga(weights):
    _, _, obj_val, discharged, sum_exit_treatment = mm.optimize_model(diagnostic_processes = diagnostic_processes, weeks = weeks, N_input = M, M_input = M, shift = weeks*7-1, with_rolling_horizon = False, in_iteration = True, weights = weights)
    return discharged + sum_exit_treatment

# Problem Definition
problem = structure()
#problem.costfunc = sphere
problem.costfunc = optimize_ga
problem.nvar = M
problem.varmin = np.zeros(M)
problem.varmax = np.full(M, 100)

# GA Parameters
params = structure()
params.maxit = 5
params.npop = 5
params.beta = 1
params.pc = 1
params.gamma = 0.1 # En høyere gamma-verdi, gir større variasjon i uniformfordelingen ved crossover.-->
params.mu = 0.3
params.sigma = 0.1

# Run GA
out = ga.run(problem, params)

# Results

plt.plot(out.bestcost)
# plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()
