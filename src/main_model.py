import gurobipy as gp
from gurobipy import GRB
import numpy as np
import model_functions
import param

#Set sizes
J=2
T=D=2
N=10
R=3

W_jn=param.W_jn
Q_ij=param.Q_ij
D_jt=param.D_jt
B_jr=param.B_jr
E_jn=param.E_jn
M_ij=param.M_ij

C_sol,Q_sol=model_functions.loadSolution(J,T,N, "/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")

patient_processes=np.matrix([
        [1,0,0,1,0,1],
        [1,1,1,0,0,1]])

number_of_queues=patient_processes.sum()
try:

    # Model setup
    m = gp.Model("mip_model")
    #m.setParam('TimeLimit', 60)

    # Variables
    q = m.addVars(J,T,N,name="q")
    c = m.addVars(J,T,N, vtype=GRB.INTEGER, name="c")

    # Objective function
    m.setObjective(gp.quicksum(q[j,t,n]*W_jn[j,n] for j in range(J) for t in range (T) for n in range(N) ), GRB.MINIMIZE)

    # Constraints

    #CONSTRAINT 2 IN HANS
    for j in range(J):
        for t in range(0,T):
            if t==0:
                m.addConstr(q[j,t,0]==D_jt[j,t]+gp.quicksum(C_sol[i,t,n]*Q_ij[i,j] for i in range(J) for n in range(N)))
            else:
                m.addConstr(q[j,t,0]==D_jt[j,t]+gp.quicksum(c[i,t-M_ij[i,j],n]*Q_ij[i,j] for i in range(J) for n in range(N)))

    #CONSTRAINT 3 IN HANS
    for j in range(J):
        for t in range(1,T):
            for n in range(1,N):
                m.addConstr(q[j,t,n] == q[j,t-1,n-1] - c[j,t-1,n-1])

    #CONSTRAINT 4 IN HANS
    for j in range(J):
        for t in range(T):
            for n in range(N):
                m.addConstr(c[j,t,n] <= q[j,t,n])
    #CONSTRAINT egendefinert
    for j in range(J):
        for t in range(T):
            for n in range(N):
                m.addConstr(c[j,t,n] <= 2)


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

m.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model.lp")
m.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")
