import gurobipy as gp
from gurobipy import GRB
import numpy as np
import model_functions
import model_parameters

#Set sizes
G=1
J=int(model_parameters.number_of_queues(G))
T=D=2
N=10
R=3

print("******* Running model with",J,"queues and",T,"time periods *******")

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

    #******************** Model setup ********************
    m = gp.Model("mip_queue_model")
    #m.setParam('TimeLimit', 60)

    #******************** Variables ********************
    #q = m.addVars(J,D,T,N,name="q")
    #c = m.addVars(J,D,T,N, vtype=GRB.INTEGER, name="c")
    #m.update()
    #variables=m.getVars()
    #c_dict,q_dict=model_functions.variables_to_dict(variables,J,D,T,N)
    c_dict={}
    q_dict={}
    for j in range(J):
        for d in range(D):
            for t in range(T):
                for n in range(N):
                    if t>=d:
                        c_dict[j,d,t,n]=m.addVar(vtype=GRB.INTEGER, name="q")
                        q_dict[j,d,t,n]=m.addVar(vtype=GRB.INTEGER, name="c")
                    else:
                        q_dict[j,d,t,n]=0
                        c_dict[j,d,t,n]=0
    m.update()
    #******************** Objective function ********************
    m.setObjective(gp.quicksum(q_dict[j,d,t,n]*W_jn[j,n] for j in range(J) for t in range (T) for d in range (D) for n in range(N) ), GRB.MINIMIZE)

    #******************** Constraints ********************
    #CONSTRAINT 2 IN HANS - updating the following queue when patients are being serviced
    for j in range(J):
        for t in range(0,T):
            for d in range(T):
                if t==0:
                    d=t
                    m.addConstr(q_dict[j,d,t,0]==D_jt[j,t])
                elif t==d:
                    m.addConstr(q_dict[j,d,t,0]==D_jt[j,t])
                elif t>d:
                    m.addConstr(q_dict[j,d,t,0]==gp.quicksum(c_dict[i,d,t-M_ij[i,j],n]*Q_ij[i,j] for i in range(J) for n in range(N)))
                else:
                    print("Hææ")

    #CONSTRAINT 3 IN HANS - Updating a queue when patients are serviced
    for j in range(J):
        for d in range(0,T):
            for t in range(1,T):
                for n in range(1,N):
                    m.addConstr(q_dict[j,d,t,n] == q_dict[j,d,t-1,n-1] - c_dict[j,d,t-1,n-1])



    #CONSTRAINT 4 IN HANS - cannot service more patients than there are patients in the queue
    for j in range(J):
        for t in range(T):
            for n in range(N):
                m.addConstr(gp.quicksum(c_dict[j,d,t,n] for d in range(T)) <= gp.quicksum(q_dict[j,d,t,n] for d in range(T)))

    #CONSTRAINT egendefinert, capper antall behandlinger
    for j in range(J):
        for t in range(T):
            m.addConstr(gp.quicksum(c_dict[j,d,t,n] for n in range(N) for d in range (T)) <= 2)

    for j in range(J):
        for n in range(1,N):
            m.addConstr(q_dict[j,0,0,n] == E_jn[j,n])



    #******************** Optimize model ********************
    print("******************")
    m.optimize()
    #for v in m.getVars():
        #if v.x>0:
            #print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')


for key in q_dict:
    value=q_dict[key]
    if isinstance(value, gp.Var):
        value=value.x
    if value>0:
        print('q%s %g' % (key, value))

for key in c_dict:
    value=c_dict[key]
    if isinstance(value, gp.Var):
        value=value.x
    if value>0:
        print('c%s %g' % (key, value))


#Haralds mac
#m.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model.lp")
#m.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")

#Haralds skole-pc
#m.write("C:/Users/hara/Code/Thesis/src/output/model.lp")
#m.write("C:/Users/hara/Code/Thesis/src/output/model_solution.sol")
