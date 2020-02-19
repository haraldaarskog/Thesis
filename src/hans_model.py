import gurobipy as gp
from gurobipy import GRB
import numpy as np


#Set sizes
J=2
T=3
N=10
R=3

B_jn=np.matrix([
        [1,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,8,9,10]])

Q_ij=np.matrix([
        [0,1],
        [0,0]])

delta_jt=np.matrix([
        [2,8,2,2,2,2,2,2,2,2],
        [2,2,2,2,2,2,2,2,2,2]])

try:

    # Model setup
    m = gp.Model("mip_model")
    #m.setParam('TimeLimit', 60)

    # Variables
    w = m.addVars(J,T,N,name="w")
    c = m.addVars(J,T,N, vtype=GRB.INTEGER, name="c")

    # Objective function
    m.setObjective(gp.quicksum(w[j,t,n]*B_jn[j,n] for j in range(J) for t in range (T) for n in range(N) ), GRB.MINIMIZE)


    # Constraints

    #CONSTRAINT 2 IN HANS
    for j in range(J):
        for t in range(0,T):
            if t==0:
                m.addConstr(w[j,t,0]==delta_jt[j,t]+gp.quicksum(c[i,t,n]*Q_ij[i,j] for i in range(J) for n in range(N)))
            else:
                m.addConstr(w[j,t,0]==delta_jt[j,t]+gp.quicksum(c[i,t-1,n]*Q_ij[i,j] for i in range(J) for n in range(N)))

    #CONSTRAINT 3 IN HANS
    for j in range(J):
        for t in range(1,T):
            for n in range(1,N):
                m.addConstr(w[j,t,n] == w[j,t-1,n-1] - c[j,t-1,n-1])

    #CONSTRAINT 4 IN HANS
    for j in range(J):
        for t in range(T):
            for n in range(N):
                m.addConstr(c[j,t,n] <= w[j,t,n])
    #CONSTRAINT egendefinert
    for j in range(J):
        for t in range(T):
            for n in range(N):
                m.addConstr(c[j,t,n] <= 2)

    #Start values
    #w[0,0,0].start=10
    #w[1,0,0].start=10
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

m.write("model.lp")
