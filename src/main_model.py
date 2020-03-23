import gurobipy as gp
from gurobipy import GRB
import numpy as np
import model_functions
import model_parameters
from prettytable import PrettyTable
from termcolor import colored, cprint
import sys

#TODO: Lage en funksjon som gjør at vi kan kjøre uavhengige runs og avhengige runs


#Set sizes
G=2
J=int(model_functions.number_of_queues(G))
T=5
N=10
M=10
A=model_functions.patient_processes.shape[1]
R=2


t = PrettyTable(['Queues','Time periods','N','M','Patient processes','Activities','Resources'])
t.add_row([J,T,N,M,G,A,R])
print(t)

model_functions.all_print_patient_processes(G)

#Number of days since the model was updated the last time
shift=1

#Loading parameters
W_jn=model_parameters.W_jn
Q_ij=model_parameters.Q_ij
D_jt=model_parameters.D_jt
M_ij=model_parameters.M_ij
H_jr=model_parameters.H_jr
L_tr=model_parameters.L_tr
F_ga=model_parameters.F_ga
K=100

#E_jnm=model_functions.old_solution("output/model_solution.sol","q",0)[:,-1,:,:]
E_jnm=model_functions.create_empty_initial_queue(J,N,M)
A_jt=model_functions.old_solution("output/model_solution.sol","b",shift)
#A_jt=np.array([
#    [2, 2, 1, 0, 0, 0, 0, 0, 0],
#    [0, 2, 2, 2, 2, 2, 0, 0, 0],
#    [0, 0, 2, 2, 2, 2, 2, 0, 0]])

try:

    #******************** Model setup ********************
    model = gp.Model("mip_queue_model")
    model.setParam("OutputFlag",0)

    #******************** Variables ********************
    c_dict={}
    q_dict={}
    b_dict={}
    x_dict={}
    u_A_dict={}
    u_B_dict={}
    for j in range(J):
        for t in range(T):
            for n in range(N):
                for m in range(M):
                    if m>=n:
                        c_dict[j,t,n,m]=model.addVar(name="c("+str(j)+","+str(t)+","+str(n)+","+str(m)+")")
                        q_dict[j,t,n,m]=model.addVar(name="q("+str(j)+","+str(t)+","+str(n)+","+str(m)+")")
                    else:
                        q_dict[j,t,n,m]=0
                        c_dict[j,t,n,m]=0


    for j in range(J):
        for t in range(T):
            b_dict[j,t]=model.addVar(name="b("+str(j)+","+str(t)+")")
            u_A_dict[j,t]=model.addVar(name="u_A("+str(j)+","+str(t)+")")
            u_B_dict[j,t]=model.addVar(name="u_B("+str(j)+","+str(t)+")")

    for t in range(T):
        for m in range(M):
            for g in range(G):
                for a in range(A):
                    x_dict[t,m,g,a]=model.addVar(name="x("+str(t)+","+str(m)+","+str(g)+","+str(a)+")",vtype=GRB.BINARY)






    #******************** Objective function ********************
    model.setObjective(gp.quicksum(W_jn[j,n]*q_dict[j,t,n,m] for j in range(J) for t in range(T) for n in range(N) for m in range(M)), GRB.MINIMIZE)


    #******************** Constraints ********************

    #CONSTRAINT 2 IN HANS - updating the following queue when patients are being serviced
    for j in range(J):
        for t in range(T):
            for m in range(M):
                # må ta hensyn til q[1,1,0,0]
                if m==0:
                    if t==0:
                        #Når m==0, legger til nyinkommende pasienter og sørger for at n,m=0
                        model.addConstr(q_dict[j,t,0,0]==D_jt[j,t])
                    else:
                        model.addConstr(q_dict[j,t,0,0]==D_jt[j,t]+gp.quicksum(c_dict[i,t-M_ij[i,j],0,0]*Q_ij[i,j] for i in range(J)))

                #Ikke interessant å se på t==0, siden m==0 håndterer innkommende psienter på denne dagen.
                #Setter resten lik 0 ved t=0. Må endres på når
                elif t==0:
                    for n in range(N):
                        if E_jnm[j,n,m]>0:
                            #model.addConstr(q_dict[j,t,n,m]==E_jnm[j,n,m])
                            model.addConstr(q_dict[j,t,n,m]==0)
                        else:
                            model.addConstr(q_dict[j,t,n,m]==0)
                elif t>=1:
                    model.addConstr(q_dict[j,t,0,m]==gp.quicksum(c_dict[i,t-M_ij[i,j],n,m]*Q_ij[i,j] for i in range(J) for n in range(N)))


    #CONSTRAINT 3 IN HANS - Updating a queue when patients are serviced
    for j in range(J):
        for t in range(1,T):
            for n in range(1,N):
                for m in range(1,M):
                    model.addConstr(q_dict[j,t,n,m] == q_dict[j,t-1,n-1,m-1] - c_dict[j,t-1,n-1,m-1])



    #CONSTRAINT 4 IN HANS - cannot service more patients than there are patients in the queue
    for j in range(J):
        for t in range(T):
            for n in range(N):
                for m in range(M):
                    model.addConstr(c_dict[j,t,n,m] <= q_dict[j,t,n,m])


    for j in range(J):
        for t in range(T):
            model.addConstr(b_dict[j,t]==gp.quicksum(c_dict[j,t,n,m] for n in range(N) for m in range(M)))

    #CONSTRAINT egendefinert, capper antall behandlinger
    for j in range(J):
        for t in range(T):
            model.addConstr(gp.quicksum(c_dict[j,t,n,m] for n in range(N) for m in range(M)) <= 2)


    for g in range(G):
        for a in range(A):
            j=model_functions.find_queue(g,a)
            if j==-1:
                continue
            for m in range(M):
                for t in range(T):
                    model.addConstr(x_dict[t,m,g,a]<=gp.quicksum(c_dict[j,t,n,m] for n in range(N)))
                    model.addConstr(10000*x_dict[t,m,g,a]>=gp.quicksum(c_dict[j,t,n,m] for n in range(N)))

    for t in range(T):
        for r in range(R):
            #model.addConstr(gp.quicksum(H_jr[j,r]*b_dict[j,t] for j in range(J)) <= L_tr[t,r])
            pass

    for t in range(T):
        for m in range(M):
            for g in range(G):
                for a in range(A):
                    #model.addConstr(m*x_dict[t,m,g,a]<=F_ga[g,a])
                    pass

    for t in range(T):
        for g in range(G):
            for a in range(A):
                j=model_functions.find_queue(g,a)
                for m in range(M):
                    if m>F_ga[g,a]:
                        for n in range(N):
                            #model.addConstr(q_dict[j,t,n,m]<=0)
                            pass
    for j in range(J):
        for t in range(T-shift):
            #model.addConstr(u_A_dict[j,t]-u_B_dict[j,t]==b_dict[j,t]-A_jt[j,t])
            pass


    model.addConstr(gp.quicksum(u_A_dict[j,t] for j in range(J) for t in range(T))<=K)
    model.addConstr(gp.quicksum(u_B_dict[j,t] for j in range(J) for t in range(T))<=K)




    #******************** Optimize model ********************
    print("\n")
    model.optimize()
    status=model.status
    if status==2:
        runtime=model.Runtime
        print(colored("Found optimal solution in %g seconds (%g minutes)" % (runtime,(runtime/60)), 'green',attrs=['bold']))
    elif status==3:
        print(colored("Model is infeasible", 'red',attrs=['bold']))
        sys.exit()
    else:
        print(colored("Check gurobi status codes", 'red',attrs=['bold']))
        sys.exit()
    print(colored('Objective value: %g' % model.objVal, 'magenta',attrs=['bold']))
    print("\n")
except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))



print(colored("q(j, t, n, m)", 'green',attrs=['underline']))
model_functions.variable_printer("q", q_dict)
print("\n")
print(colored("c(j, t, n, m)", 'green',attrs=['underline']))
model_functions.variable_printer("c",c_dict)
print("\n")
print(colored("b(j, t)", 'green',attrs=['underline']))
model_functions.variable_printer("b",b_dict)
print("\n")
print(colored("x(t, m, g, a)", 'green',attrs=['underline']))
model_functions.variable_printer("x",x_dict)
print("\n")
print(colored("u_A(j,t)", 'green',attrs=['underline']))
model_functions.variable_printer("u_A",u_A_dict)
print("\n")
print(colored("u_B(j,t)", 'green',attrs=['underline']))
model_functions.variable_printer("u_B", u_B_dict)


#Haralds mac
model.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model.lp")
model.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")
