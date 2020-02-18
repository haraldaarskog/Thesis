import gurobipy as gp
from gurobipy import GRB
import numpy as np


#Set sizes
J=2
T=2
N=10
R=3

J_r=1

B_jn=np.matrix([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])

Q_ij=np.matrix([[1,1],[1,1]])

delta_jt=np.matrix([[5,5],[5,5]])

try:

    # Create a new model
    m = gp.Model("mip1")
    #m.setParam('TimeLimit', 60)
    # Create variables
    z = m.addVar(vtype=GRB.BINARY, name="z")
    w = m.addVars(J,T,N,name="w")
    c = m.addVars(J,T,N, vtype=GRB.INTEGER, name="c")

    # Set objective
    m.setObjective(sum(w[j,t,n]*B_jn[j,n] for j in range(J) for t in range (T) for n in range(N) ), GRB.MINIMIZE)

 
    #for j in range(0,J):
        #for n in range(N):
            #m.addConstr(sum(w[j,t,n] for t in range(0,T))>= 50, "constraint 1")
        
    for j in range(J):
        for t in range(T):          
            m.addConstr(w[j,t,0]==delta_jt[j,t]+sum(c[i,t,n]*Q_ij[i,j] for i in range(J)))

    for j in range(J):
        for t in range(1,T):
            for n in range(1,N):
                m.addConstr(w[j,t,n] == w[j,t-1,n-1] - c[j,t-1,n-1])
                
    for j in range(J):
        for t in range(T):
            for n in range(N):
                m.addConstr(c[j,t,n] <= w[j,t,n])

    w[0,0,0].start=10
    w[1,0,0].start=10
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