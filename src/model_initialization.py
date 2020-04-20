import gurobipy as gp
from gurobipy import GRB
import numpy as np
from termcolor import colored, cprint
import sys
import model_functions as mf
import model_parameters as mp


#A function that runs the model
#If with_rolling_horizon = True, input from the previous run is included
#If reset = True, input from previous run is deleted
#shift: number of days between running the model the last time and today. The number
#of days that are already implemented
def optimize_model(weeks, N, M, with_rolling_horizon, reset, obj_weights, shift):
    #********************Set sizes********************
    Patient_processes = 5


    total_diagnosis_queues = mf.get_total_number_of_diagnosis_queues()
    total_treatment_queues = mf.get_total_number_of_treatment_queues()

    total_queues = mf.get_total_number_of_queues()

    #The number of queues is a function of the number of patient processes
    #and the number of activities in each patient process
    Queues = mf.number_of_queues(Patient_processes)

    number_of_current_queues = Queues + total_treatment_queues

    set_of_last_queues_in_diagnosis=mf.generate_last_queues_in_diagnosis()

    Time_periods = 7 * weeks

    Activities = mp.patient_processes.shape[1]
    Resources = mp.activity_resource_map.shape[1]

    if shift >= Time_periods:
        print(colored("Shift must be less than the number of time periods", 'red', attrs = ['bold']))
        sys.exit()


    #********************Loading parameters********************
    D_jt = mp.D_jt
    M_ij = mf.create_M_ij()
    H_jr = mp.H_jr
    L_rt = mp.L_rt
    F_ga = mp.F_ga
    K = mp.K
    Q_ij = mf.create_Q_ij()



    #if the model is executed with RH, the input from previous runs are included
    if with_rolling_horizon == True and reset == False:
        A_jt = mf.old_solution("output/model_solution.sol", "b", shift)
        E_jnm = mf.create_E_jnm(total_queues, N, M, shift)
        c_jtnm_old = mf.old_solution("output/model_solution.sol", "c", 0)
        G_jtnm = mf.serviced_in_previous(total_queues, Time_periods, N, M, shift, c_jtnm_old, Q_ij, M_ij)
    else:
        E_jnm = np.zeros((total_queues, N, M))
        A_jt = np.zeros((total_queues, Time_periods))
        G_jtnm = np.zeros((total_queues, Time_periods, N, M))
    try:
        #******************** Model setup ********************

        model  =  gp.Model("mip_queue_model")
        model.setParam("OutputFlag", 0)

        #******************** Variables ********************


        c_dict  =  model.addVars(total_queues, Time_periods, N, M, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "c")
        q_dict  =  model.addVars(total_queues, Time_periods, N, M, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "q")

        b_dict = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "b")
        w_dict = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "w")

        #x_dict = model.addVars(Time_periods, M, Patient_processes, Activities, vtype = GRB.BINARY, lb = 0.0, ub = GRB.INFINITY, name = "x")

        u_A_dict = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "u_A")
        u_B_dict = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "u_B")

        model.update()
        for j in range(total_queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        if n>m:
                            model.remove(c_dict[j, t, n, m])
                            model.remove(q_dict[j, t, n, m])

        for j in range(Queues, total_queues - total_treatment_queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        model.remove(c_dict[j, t, n, m])
                        model.remove(q_dict[j, t, n, m])

        #******************** Objective function ********************
        model.setObjective(gp.quicksum(obj_weights[j, n] * q_dict[j, t, n, m]  for j in range(total_queues) for t in range(Time_periods) for n in range(N) for m in range(M)), GRB.MINIMIZE)



        #******************** Constraints ********************

        #Updating queues whith serviced patients/patients. Denne restriksjonen tar lang tid å generere.

        for j in range(total_queues):
            if j < Queues:
                for t in range(Time_periods):
                    for m in range(M):
                        if m == 0:
                            model.addConstr(q_dict[j, t, 0, 0] == D_jt[j, t % 7] + gp.quicksum(G_jtnm[j, t, n, 0] for n in range(N)) + gp.quicksum(c_dict[i, t - M_ij[i], 0, 0] * Q_ij[i, j] for i in range(Queues) if (t - M_ij[i]) >= 0))
                        elif t == 0 and m >= 1:
                            for n in range(N):
                                if n <= m:
                                    model.addConstr(q_dict[j, t, n, m] == E_jnm[j, n, m] + G_jtnm[j, t, n, m])
                        else:
                            model.addConstr(q_dict[j, t, 0, m] == gp.quicksum(G_jtnm[j, t, n, m] for n in range(N)) + gp.quicksum(c_dict[i, t - M_ij[i], n, m] * Q_ij[i, j] for i in range(Queues) for n in range(N) if (t - M_ij[i]) >= 0))
            elif j >= total_queues - total_treatment_queues:
                for t in range(Time_periods):
                    for m in range(M):
                        if t == 0 and m >= 1:
                            for n in range(N):
                                model.addConstr(q_dict[j, t, n, m] == E_jnm[j, n, m] + G_jtnm[j, t, n, m])
                        elif mf.is_first_queue_in_treatment(j):
                            model.addConstr(q_dict[j, t, 0, m] == mp.share_of_patients_into_treatment * (1 / mf.get_number_of_treatment_paths()) * gp.quicksum(c_dict[i, t - M_ij[i], n, m] for i in set_of_last_queues_in_diagnosis for n in range(N) if (t - M_ij[i]) >= 0))
                        else:
                            model.addConstr(q_dict[j, t, 0, m] == gp.quicksum(G_jtnm[j, t, n, m] for n in range(N)) + gp.quicksum(c_dict[i, t - M_ij[i], n, m] * Q_ij[i, j] for i in range(total_treatment_queues, total_queues) for n in range(N) if (t - M_ij[i]) >= 0))

            else:
                continue

        #Updating a queue when patients are serviced
        for j in range(total_queues):
            for t in range(1, Time_periods):
                for n in range(1, N):
                    for m in range(1, M):
                        model.addConstr(q_dict[j, t, n, m] == q_dict[j, t - 1, n - 1, m - 1] - c_dict[j, t - 1, n - 1, m - 1])

        #Constriants ensuring that the number of serviced patients are equal to or less than then number of patients in the queue
        for j in range(total_queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        model.addConstr(c_dict[j, t, n, m] <= q_dict[j, t, n, m])

        #Defining b_jt
        for j in range(total_queues):
            for t in range(Time_periods):
                model.addConstr(b_dict[j, t] == gp.quicksum(c_dict[j, t, n, m] for n in range(N) for m in range(M)))


        for j in range(total_queues):
            for t in range(Time_periods):
                model.addConstr(w_dict[j, t] == gp.quicksum(q_dict[j, t, n, m] for n in range(N) for m in range(M)))

        #Resource constraints
        for t in range(Time_periods):
            for r in range(Resources):
                model.addConstr(gp.quicksum(H_jr[j, r] * b_dict[j, t] for j in range(total_queues)) <=  L_rt[r, t % 7])




        #Shifting constraints
        for j in range(total_queues):
            for t in range(Time_periods - shift):
                model.addConstr(u_A_dict[j, t] - u_B_dict[j, t] == b_dict[j, t] - A_jt[j, t])
                pass

        #The number of shifts must be below a value K
        model.addConstr(gp.quicksum(u_A_dict[j, t] for j in range(total_queues) for t in range(Time_periods)) <= K)
        model.addConstr(gp.quicksum(u_B_dict[j, t] for j in range(total_queues) for t in range(Time_periods)) <= K)

        #******************** Optimize model ********************
        model.optimize()

        status = model.status
        if status == 2:
            runtime = model.Runtime
            print(colored("Found optimal solution in %g seconds (%g minutes)" % (runtime, (runtime / 60)), 'green', attrs = ['bold']))
        elif status == 3:
            print(colored("Model is infeasible", 'red', attrs = ['bold']))
            sys.exit()
        else:
            print(colored("Check gurobi status codes", 'red', attrs = ['bold']))
            sys.exit()
        print(colored('Objective value: %g' % model.objVal, 'magenta', attrs = ['bold']))
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    b_dict = createe(b_dict)
    q_dict = createe(q_dict)
    return q_dict, b_dict



#Running the model
def run_model():
    q_dict, c_dict = optimize_model(weeks = 1, shift = 6, with_rolling_horizon = True, reset = True)



if __name__ == '__main__':
    run_model()

def createe(d):
    new_d={}
    for key in d:
        try:
            new_d[key] = d[key].x
        except Exception as e:
            continue
    return new_d
