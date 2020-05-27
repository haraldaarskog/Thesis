import gurobipy as gp
from gurobipy import GRB
import numpy as np
from beautifultable import BeautifulTable
from termcolor import colored, cprint
import sys
import time
from datetime import datetime
import model_gui as mg
import model_functions as mf
import model_parameters as mp
import model_obj_function as mof
import model_printer as mop

overall_start = time.time()
sol_file_name = "output/model_solution.sol"

#A function that runs the model
#If with_rolling_horizon = True, input from the previous run is included
#shift: number of days between running the model the last time and today
def optimize_model(weeks, N_input, M_input, shift, with_rolling_horizon, in_iteration, weights, E, G):
    start_initialization_time = time.time()



    #********************Set sizes********************
    Patient_processes = mp.diagnostic_processes.shape[0]

    #Different queue sizes
    total_diagnosis_queues = mf.get_total_number_of_diagnosis_queues()
    total_treatment_queues = mf.get_total_number_of_treatment_queues()
    total_queues = mf.get_total_number_of_queues()


    #The queues marking the end of the diagnosis phase in a patient process
    set_of_last_queues_in_diagnosis = mf.generate_last_queues_in_diagnosis()

    week_length = mp.week_length
    Time_periods = week_length * weeks
    N = N_input
    M = M_input

    #The number of activities is dervied from the number of columns in the patient process matrix
    Activities = len(mp.activity_dict.keys())
    #The number of resources is dervied from the number of columns in the activity-resource matrix
    Resources = mp.L_rt.shape[0]

    #The shift can not be larger than the number of time periods.
    if shift >= Time_periods:
        print(colored("Shift must be less than the number of time periods", 'red', attrs = ['bold']))
        sys.exit()

    #********************Information prints********************

    if not in_iteration == True:
        mop.create_set_table(total_queues, Time_periods, N, M, Patient_processes, Activities, Resources)
        print("Diagnostic paths")
        mop.all_print_patient_processes(Patient_processes)
        print("Treatment paths")
        mop.print_treatment_processes()

    #********************Loading parameters********************

    Patient_arrivals_jt = mp.Patient_arrivals_jt
    M_j = mf.create_M_j()
    L_rt = mp.L_rt
    Time_limits_j = mp.Time_limits_j
    K_t = mf.create_K_parameter(start_value = 10, increase_per_week = 0.5, time_periods = Time_periods)
    Q_ij = mf.create_Q_ij()
    queue_to_path = mf.create_queue_to_path(total_queues)
    probability_of_path = mp.probability_of_path
    H_jr = mf.create_H_jr()

    #if the model is executed with RH, the input from previous runs are included
    #otherwise, the input from previous runs are all zero.
    if with_rolling_horizon == True:
        A_jt = mf.old_solution("output/model_solution.sol", "b", shift)
        #E_jnm = mf.create_E_jnm(total_queues, N, M, shift, sol_file_name)
        c_jtnm_old = mf.old_solution("output/model_solution.sol", "c", 0)
        #G_jtnm = mf.calculate_rollover_service(total_queues, Time_periods, N, M, shift, c_jtnm_old, Q_ij, M_j)
        if E is None or G is None:
            E_jnm = np.zeros((total_queues, N, M))
            G_jtm = np.zeros((total_queues, Time_periods, M))
        else:
            E_jnm = mf.from_dict_to_matrix_2(E, (total_queues, N, M))
            G_jtm = mf.from_dict_to_matrix_2(G, (total_queues,Time_periods,M))

    else:
        E_jnm = np.zeros((total_queues, N, M))
        A_jt = np.zeros((total_queues, Time_periods))
        G_jtm = np.zeros((total_queues, Time_periods, M))


    #printing the initialization time
    end_initialization_time = time.time()
    if not in_iteration == True:
        print("Initialization time: ", end_initialization_time-start_initialization_time, "seconds")

    try:
        #******************** Model setup ********************

        #initializing the model
        model = gp.Model("mip_queue_model")

        #surpressing gurobi output
        #model.setParam("OutputFlag", 0)

        #******************** Variables ********************
        start_variables = time.time()

        c_variable  =  model.addVars(total_queues, Time_periods, N, M, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "c")
        q_variable  =  model.addVars(total_queues, Time_periods, N, M, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "q")

        b_variable = model.addVars(total_queues, Time_periods, vtype = GRB.INTEGER, lb = 0.0, ub = GRB.INFINITY, name = "b")

        u_A_variable = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "u_A")
        u_R_variable = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "u_R")

        model.update()

        #removing variables where n>m
        for j in range(total_queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        if n > m:
                            model.remove(c_variable[j, t, n, m])
                            model.remove(q_variable[j, t, n, m])

        end_variables = time.time()
        if not in_iteration == True:
            print("Generating variables:", end_variables - start_variables)
        #******************** Objective function ********************
        model.setObjective(gp.quicksum(mof.obj_weights_m(j, m) * q_variable[j, t, n, m]  for j in range(total_queues) for t in range(Time_periods) for n in range(N) for m in range(M)), GRB.MINIMIZE)

        #******************** Constraints ********************

        start_constraints = time.time()

        #Updating queues whith serviced patients
        for j in range(total_queues):
            if j < total_diagnosis_queues:
                for t in range(Time_periods):
                    for m in range(M):
                        if m == 0:
                            model.addConstr(q_variable[j, t, 0, 0] == Patient_arrivals_jt[j, t % week_length] + G_jtm[j, t, 0] + gp.quicksum(c_variable[i, t - M_j[i], 0, 0] * Q_ij[i, j] for i in range(total_diagnosis_queues) if (t - M_j[i]) >= 0))
                        elif t == 0 and m > 0:
                            for n in range(N):
                                if n <= m:
                                    model.addConstr(q_variable[j, t, n, m] == E_jnm[j, n, m])
                        else:
                            model.addConstr(q_variable[j, t, 0, m] == G_jtm[j, t, m] + gp.quicksum(c_variable[i, t - M_j[i], n, m] * Q_ij[i, j] for i in range(total_diagnosis_queues) for n in range(N) if (t - M_j[i]) >= 0))


            else:
                for t in range(Time_periods):
                    for m in range(M):
                        if t == 0 and m > 0:
                            for n in range(N):
                                model.addConstr(q_variable[j, t, n, m] == E_jnm[j, n, m])
                        elif mf.is_first_queue_in_treatment(j):
                            g_j = queue_to_path[j]
                            model.addConstr(q_variable[j, t, 0, m] == G_jtm[j, t, m] + gp.quicksum(probability_of_path[queue_to_path[i], g_j] * c_variable[i, t - M_j[i], n, m] for i in set_of_last_queues_in_diagnosis for n in range(N) if (t - M_j[i]) >= 0))
                        else:
                            model.addConstr(q_variable[j, t, 0, m] == G_jtm[j, t, m] + gp.quicksum(c_variable[i, t - M_j[i], n, m] * Q_ij[i, j] for i in range(total_diagnosis_queues, total_queues) for n in range(N) if (t - M_j[i]) >= 0))


        #print("Cons1 total:", time.time()-start_constraints)



        #Updating a queue when patients are serviced
        for j in range(total_queues):
            for t in range(1, Time_periods):
                for n in range(1, N):
                    for m in range(1, M):
                        model.addConstr(q_variable[j, t, n, m] == q_variable[j, t - 1, n - 1, m - 1] - c_variable[j, t - 1, n - 1, m - 1])

        #Constriants ensuring that the number of serviced patients are equal to or less than then number of patients in the queue
        for j in range(total_queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        model.addConstr(c_variable[j, t, n, m] <= q_variable[j, t, n, m])

        #Defining b_jt
        for j in range(total_queues):
            for t in range(Time_periods):
                model.addConstr(b_variable[j, t] == gp.quicksum(c_variable[j, t, n, m] for n in range(N) for m in range(M)))


        #Resource constraints
        for r in range(Resources):
            for t in range(Time_periods):
                model.addConstr(gp.quicksum(H_jr[j, r] * b_variable[j, t] for j in range(total_queues)) <=  L_rt[r, t % week_length])


        #Time limit constraints
        for j in range(total_queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        if m > Time_limits_j[j]:
                            model.addConstr(q_variable[j, t, n, m] <= 0)


        #Shifting constraints
        #rint(A_jt.shape)
        for j in range(total_queues):
            for t in range(Time_periods - shift - 1):
                model.addConstr(u_A_variable[j, t] - u_R_variable[j, t] == b_variable[j, t] - A_jt[j, t])


        #The number of shifts must be below a value K
        for t in range(Time_periods):
            model.addConstr(gp.quicksum(u_A_variable[j, t] for j in range(total_queues)) <= K_t[t])
            model.addConstr(gp.quicksum(u_R_variable[j, t] for j in range(total_queues)) <= K_t[t])

        #Printing the time it took to generate the constraints
        end_constraints = time.time()
        if not in_iteration == True:
            print("Generating constraints:", end_constraints - start_constraints)
            print("\n")
            print("Optimizing model...")
        #******************** Optimize model ********************

        model.optimize()
        status = model.status
        runtime = model.runtime
        mop.print_model_status(status, runtime)
        objective_value = model.objVal

        print(colored('Objective value: %g' % objective_value, 'magenta', attrs = ['bold']))

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))


    #sum_exit_diagnosis, sum_exit_treatment = mf.number_of_exit_patients(c_variable, shift)
    #discharged = mf.calculate_discharged_patients(c_variable)
    if not in_iteration == True:
        start_print = time.time()
        mop.print_variables(q_variable, c_variable, b_variable, u_A_variable, u_R_variable)

        model.write("output/model.lp")
        model.write(sol_file_name)

        number_of_constraints = len(model.getConstrs())

        number_of_variables = len(model.getVars())


        sum_D, sum_E, sum_G= mf.calculate_stats(total_diagnosis_queues, Patient_arrivals_jt, E_jnm, G_jtm, shift)

        new_patient_arrivals = sum_D * weeks

        #Denne tar lang tid
        #rollover_queue_next_period = np.sum(mf.create_E_jnm(total_queues, N, M, shift, sol_file_name))

        #rollover_queue_next_period = np.around(rollover_queue_next_period, decimals=2)

        #denne tar lang tid
        #c_jtnm_current = mf.old_solution(sol_file_name, "c", 0)

        #tar noe tid
        #arr = mf.calculate_rollover_service(total_queues, Time_periods, N, M, shift, c_jtnm_current, Q_ij, M_j)

        #tar noe tid
        #rollover_service_next_period = np.sum(mf.calculate_rollover_service(total_queues, Time_periods, N, M, shift, c_jtnm_current, Q_ij, M_j))
        #rollover_service_next_period = np.around(rollover_service_next_period, decimals=2)



        mop.create_overview_table(new_patient_arrivals, sum_E, sum_G)#, sum_exit_diagnosis, sum_exit_treatment, rollover_queue_next_period, rollover_service_next_period, discharged)
        #Outputing the time it took to print results and write to file
        end_print = time.time()
        print("Printing:", end_print - start_print)
        #mg.create_gantt_chart(total_queues, Time_periods, mf.loadSolution(sol_file_name)["b"])

    b_variable = mf.convert_dict(b_variable)
    q_variable = mf.convert_dict(q_variable)
    #mop.print_resource_utilization(total_queues, Time_periods,b_variable)
    #print(discharged + sum_exit_treatment)
    total_elapsed_time = time.time() - overall_start
    #tar noe tid
    if not in_iteration == True:
        mf.write_to_file(total_queues, Time_periods, N, M, objective_value, number_of_variables, number_of_constraints, runtime, total_elapsed_time)
    return q_variable, b_variable, objective_value


#Running the model
def run_model():
    w = 2
    optimize_model(weeks = w, N_input = 20, M_input = 20, shift = 6, with_rolling_horizon = False, in_iteration = False, weights = None, E = None, G = None)

if __name__ == '__main__':
    run_model()
