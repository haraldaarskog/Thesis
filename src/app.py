import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import model_functions as mf
import main_model as mm

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
problem.varmax = np.full(M, 1)

# GA Parameters
params = structure()
params.maxit = 10
params.npop = 2
params.mu = 0.5
params.sigma = 0.2
#the number of pairs of children to generate each iteration
params.nc = int(np.round(params.npop/2)*2)//2

# Run GA
out = ga.run(problem, params)

# Results
y = out.ch
y_b = out.ch_b
y = np.insert(y, 0, out.bestcost[0])
y_b = np.insert(y_b, 0, out.bestcost[0])
#x = np.arange(0, params.maxit + params.maxit/len(out.ch), step=params.maxit/len(out.ch))
x = np.arange(0, len(y))

plt.plot(x, y, linestyle='-', color='b', label='Value of children')
plt.plot(x, y_b, linestyle='--', color='r', label='Best value')
plt.xlim(0, params.maxit)
plt.xlabel('Children generated')
plt.ylabel('Values')
plt.legend()
plt.title('Genetic Algorithm')
plt.grid(True)
plt.savefig('figures/Score_of_all_children.png')
