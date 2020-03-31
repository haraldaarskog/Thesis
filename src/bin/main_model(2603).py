import gurobipy as gp
from gurobipy import GRB
import numpy as np
import model_functions as mf
import model_parameters as mp
from prettytable import PrettyTable
from termcolor import colored, cprint
import sys
import time
from datetime import datetime

#TODO: Lage en funksjon som gjør at vi kan kjøre uavhengige runs og avhengige runs

start_time=time.time()

#if with_rolling_horizon=True, input from the previous run is included

def run_model(with_rolling_horizon, weeks):
    #Set sizes
    G=2
    J=int(mf.number_of_queues(G))
    T=7*weeks
    N=100
    M=100
    A=mf.patient_processes.shape[1]
    R=5

    t = PrettyTable(['Queues','Time periods','N','M','Patient processes','Activities','Resources'])
    t.add_row([J,T,N,M,G,A,R])
    print(t)

    mf.all_print_patient_processes(G)

    #Number of days since the model was updated the last time
    shift=1

    #Loading parameters
    #W_jn=mp.W_jn
    Q_ij=mp.Q_ij
    D_jt=mp.D_jt
    M_ij=mp.M_ij
    H_jr=mp.H_jr
    L_tr=mp.L_rt
    F_ga=mp.F_ga
    K=100

    if with_rolling_horizon==True:
        E_jnm=mf.old_solution("output/model_solution.sol","q",0)[:,-1,:,:]
        print(E_jnm)
        A_jt=mf.old_solution("output/model_solution.sol","b",shift)
        c_jtnm_old=mf.old_solution("output/model_solution.sol","c",0)
        G_jtnm=mf.serviced_in_previous(J,T,N,M,T,c_jtnm_new)
    else:
        E_jnm=mf.create_empty_initial_queue(J,N,M)
        A_jt=np.zeros((J,T))
        G_jtnm=np.zeros((J,T,N,M))
        #G_jtnm[1,0,0,0]=0


    end_time=time.time()
    print("Initialization took", end_time-start_time,"seconds")
    try:

        #******************** Model setup ********************
        model = gp.Model("mip_queue_model")
        #model.setParam("OutputFlag",0)

        #******************** Variables ********************

        start_variables=time.time()
        c_dict=model.addVars(J,T,N,M)
        q_dict=model.addVars(J,T,N,M)
        b_dict=model.addVars(J,T)
        x_dict=model.addVars(T,M,G,A)
        u_A_dict=model.addVars(J,T)
        u_B_dict=model.addVars(J,T)
        for j in range(J):
            for t in range(T):
                for n in range(N):
                    for m in range(M):
                        if n>m:
                            model.remove(c_dict[j,t,n,m])
                            model.remove(q_dict[j,t,n,m])
        """
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
                        x_dict[t,m,g,a]=model.addVar(name="x("+str(t)+","+str(m)+","+str(g)+","+str(a)+")", vtype=GRB.BINARY)




        """
        end_variables=time.time()
        print("Generating variables:", end_variables-start_variables)
        #******************** Objective function ********************

        model.setObjective(gp.quicksum(mp.obj_weights(n)*q_dict[j,t,n,m] for j in range(J) for t in range(T) for n in range(N) for m in range(M)), GRB.MINIMIZE)


        #******************** Constraints ********************
        start_constraints=time.time()

        for j in range(J):
            for t in range(T):
                for m in range(M):
                    if m==0:
                        model.addConstr(q_dict[j,t,0,0]==D_jt[j,t%7]+gp.quicksum(G_jtnm[j,t,n,0] for n in range(N))+gp.quicksum(c_dict[i,t-M_ij[i,j],0,0]*Q_ij[i,j] for i in range(J) if (t-M_ij[i,j])>=0))
                    elif t==0 and m>=1:
                        for n in range(N):
                            if n<=m:
                                model.addConstr(q_dict[j,t,n,m]==E_jnm[j,n,m]+G_jtnm[j,t,n,m])
                    else:
                        model.addConstr(q_dict[j,t,0,m]==gp.quicksum(G_jtnm[j,t,n,m] for n in range(N))+gp.quicksum(c_dict[i,t-M_ij[i,j],n,m]*Q_ij[i,j] for i in range(J) for n in range(N) if (t-M_ij[i,j])>=0))


        #Updating a queue when patients are serviced
        for j in range(J):
            for t in range(1,T):
                for n in range(1,N):
                    for m in range(1,M):
                        #if j==0 and n==1 and m==1 and t ==1:
                            #model.addConstr(q_dict[j,t,n,m]==3)
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
        #for j in range(J):
        #    for t in range(T):
        model.addConstrs(gp.quicksum(c_dict[j,t,n,m] for n in range(N) for m in range(M)) <= 2 for j in range(J) for t in range(T))


        #TODO: Finn verdi for Big S
        for g in range(G):
            for a in range(A):
                j=mf.find_queue(g,a)
                if j==-1:
                    continue
                for m in range(M):
                    for t in range(T):
                        model.addConstr(x_dict[t,m,g,a]<=gp.quicksum(c_dict[j,t,n,m] for n in range(N)))
                        model.addConstr(10000*x_dict[t,m,g,a]>=gp.quicksum(c_dict[j,t,n,m] for n in range(N)))

        for t in range(T):
            for r in range(R):
                #model.addConstr(gp.quicksum(H_jr[j,r]*b_dict[j,t] for j in range(J)) <= L_rt[r,t%7])
                pass

        for t in range(T):
            for m in range(M):
                for g in range(G):
                    for a in range(A):
                        #model.addConstr(m*x_dict[t,m,g,a]<=F_ga[g,a])
                        pass

        #TODO: endre settet A til et sett A hvor aktivitenene er "fase-sluttende" aktiviteter
        for t in range(T):
            for g in range(G):
                for a in range(A):
                    j=mf.find_queue(g,a)
                    for m in range(M):
                        if m>F_ga[g,a]:
                            for n in range(N):
                                #model.addConstr(q_dict[j,t,n,m]<=0)
                                pass
        for j in range(J):
            for t in range(T-shift):
                model.addConstr(u_A_dict[j,t]-u_B_dict[j,t]==b_dict[j,t]-A_jt[j,t])
                pass


        model.addConstr(gp.quicksum(u_A_dict[j,t] for j in range(J) for t in range(T))<=K)
        model.addConstr(gp.quicksum(u_B_dict[j,t] for j in range(J) for t in range(T))<=K)



        end_constraints=time.time()
        print("Generating constraints:",end_constraints-start_constraints)
        #******************** Optimize model ********************
        start_opti=time.time()
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
        end_opti=time.time()
        print("Opti time:", end_opti-start_opti)


    start_print=time.time()
    print(colored("q(j, t, n, m)", 'green',attrs=['underline']))
    mf.variable_printer("q", q_dict)
    print("\n")
    print(colored("c(j, t, n, m)", 'green',attrs=['underline']))
    mf.variable_printer("c",c_dict)
    print("\n")
    print(colored("b(j, t)", 'green',attrs=['underline']))
    mf.variable_printer("b",b_dict)
    print("\n")
    print(colored("x(t, m, g, a)", 'green',attrs=['underline']))
    mf.variable_printer("x",x_dict)
    #print("\n")
    #print(colored("u_A(j,t)", 'green',attrs=['underline']))
    #mf.variable_printer("u_A",u_A_dict)
    #print("\n")
    #print(colored("u_B(j,t)", 'green',attrs=['underline']))
    #mf.variable_printer("u_B", u_B_dict)


    #Haralds mac
    model.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model.lp")
    model.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")


    constrs = model.getConstrs()
    number_of_constraints=len(constrs)
    vars=model.getVars()
    number_of_variables=len(vars)
    mf.write_to_file(J,T,N,M,G,A,R,model.objVal,number_of_variables,number_of_constraints,runtime)
    end_print=time.time()
    print("Printing:", end_print-start_print)


run_model(False, 1)
