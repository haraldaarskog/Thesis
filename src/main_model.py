import gurobipy as gp
from gurobipy import GRB
import numpy as np
import model_functions
import model_parameters

#Set sizes
G=2
J=int(model_parameters.number_of_queues(G))
T=D=2
N=10
R=3

print("Running model with",J,"queues")

#Loading parameters
W_jn=model_parameters.W_jn
Q_ij=model_parameters.Q_ij
D_jt=model_parameters.D_jt
B_jr=model_parameters.B_jr
E_jn=model_parameters.E_jn
M_ij=model_parameters.M_ij


#Loading variables from previous solution

#Haralds mac
#C_sol,Q_sol=model_functions.loadSolution(J,T,N, "/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")

#Haralds skole-pc
#C_sol,Q_sol=model_functions.loadSolution(J,T,N, "C:/Users/hara/Code/Thesis/src/output/model_solution.sol")


try:

    # Model setup
    m = gp.Model("mip_queue_model")
    #m.setParam('TimeLimit', 60)

    # Variables
    q = m.addVars(J,T,N,name="q")
    c = m.addVars(J,T,N, vtype=GRB.INTEGER, name="c")

    # Objective function
    m.setObjective(gp.quicksum(q[j,t,n]*W_jn[j,n] for j in range(J) for t in range (T) for n in range(N) ), GRB.MINIMIZE)

    # Constraints

    #CONSTRAINT 2 IN HANS - updating the following queue when patients are being serviced
    for j in range(J):
        for t in range(0,T):
            if t==0:
                m.addConstr(q[j,t,0]==D_jt[j,t])
            else:
                m.addConstr(q[j,t,0]==D_jt[j,t]+gp.quicksum(c[i,t-M_ij[i,j],n]*Q_ij[i,j] for i in range(J) for n in range(N)))

    #CONSTRAINT 3 IN HANS - Updating a queue when patients are serviced
    for j in range(J):
        for t in range(1,T):
            for n in range(1,N):
                m.addConstr(q[j,t,n] == q[j,t-1,n-1] - c[j,t-1,n-1])

    #CONSTRAINT 4 IN HANS - cannot service more patients than there are patients in the queue
    for j in range(J):
        for t in range(T):
            for n in range(N):
                m.addConstr(c[j,t,n] <= q[j,t,n])
                
    #CONSTRAINT egendefinert, capper antall behandlinger
    for j in range(J):
        for t in range(T):
            m.addConstr(gp.quicksum(c[j,t,n] for n in range(N)) <= 2)


    for j in range(J):
        for n in range(1,N):
            m.addConstr(q[j,0,n] == E_jn[j,n])


    # Optimize model
    m.optimize()
    for v in m.getVars():
        if v.x>0:
            print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')


#Haralds mac
#m.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model.lp")
#m.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")

#Haralds skole-pc
m.write("C:/Users/hara/Code/Thesis/src/output/model.lp")
m.write("C:/Users/hara/Code/Thesis/src/output/model_solution.sol")

