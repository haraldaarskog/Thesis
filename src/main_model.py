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

start_initialization_time = time.time()

#A function that runs the model
#If with_rolling_horizon = True, input from the previous run is included
#If reset = True, input from previous run is deleted
#shift: number of days between running the model the last time and today. The number
#of days that are already implemented
def optimize_model(patient_processes, weeks, N_input, M_input, shift, with_rolling_horizon, in_iteration, weights):


    #********************Set sizes********************
    Patient_processes = patient_processes

    #Different queue sizes
    total_diagnosis_queues = mf.get_total_number_of_diagnosis_queues()
    total_treatment_queues = mf.get_total_number_of_treatment_queues()
    total_queues = mf.get_total_number_of_queues()

    #The number of diagnosis queues is a function of the number of patient processes
    #and the number of activities in each patient process
    current_diagnosis_queues = mf.number_of_queues(Patient_processes)
    number_of_current_queues = current_diagnosis_queues + total_treatment_queues

    #The queues marking the end of the diagnosis phase in a patient process
    set_of_last_queues_in_diagnosis=mf.generate_last_queues_in_diagnosis()

    Time_periods = 7 * weeks
    N = N_input
    M = M_input

    #The number of activities is dervied from the number of columns in the patient process matrix
    Activities = mp.patient_processes.shape[1]
    #The number of resources is dervied from the number of columns in the activity-resource matrix
    Resources = mp.activity_resource_map.shape[1]

    #The shift can not be larger than the number of time periods.
    if shift >= Time_periods:
        print(colored("Shift must be less than the number of time periods", 'red', attrs = ['bold']))
        sys.exit()

    #********************Information prints********************
    if not in_iteration == True:
        t = BeautifulTable()
        t.column_headers=['Queues', 'Time periods', 'N', 'M', 'Patient processes', 'Activities', 'Resources']
        t.append_row([number_of_current_queues, Time_periods, N, M, Patient_processes, Activities, Resources])
        t.set_style(BeautifulTable.STYLE_BOX)
        print(t)
        mf.all_print_patient_processes(Patient_processes)


    #********************Loading parameters********************
    Patient_arrivals_jt = mp.Patient_arrivals_jt
    M_j = mf.create_M_j()
    H_jr = mp.H_jr
    L_rt = mp.L_rt
    Time_limits_ga = mp.Time_limits_ga
    K = mp.K
    Q_ij = mf.create_Q_ij()

    #if the model is executed with RH, the input from previous runs are included
    #otherwise, the input from previous runs are all zero.
    if with_rolling_horizon == True:
        A_jt = mf.old_solution("output/model_solution.sol", "b", shift)
        E_jnm = mf.create_E_jnm(total_queues, N, M, shift)
        c_jtnm_old = mf.old_solution("output/model_solution.sol", "c", 0)
        G_jtnm = mf.serviced_in_previous(total_queues, Time_periods, N, M, shift, c_jtnm_old, Q_ij, M_j)
    else:
        E_jnm = np.zeros((total_queues, N, M))
        A_jt = np.zeros((total_queues, Time_periods))
        G_jtnm = np.zeros((total_queues, Time_periods, N, M))


    #printing the initialization time
    end_initialization_time = time.time()
    if not in_iteration == True:
        print("Initialization time: ", end_initialization_time-start_initialization_time, "seconds")

    try:
        #******************** Model setup ********************

        #initializing the model
        model = gp.Model("mip_queue_model")

        #surpressing gurobi output
        model.setParam("OutputFlag", 0)

        #******************** Variables ********************
        start_variables = time.time()

        c_variable  =  model.addVars(total_queues, Time_periods, N, M, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "c")
        q_variable  =  model.addVars(total_queues, Time_periods, N, M, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "q")

        b_variable = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "b")
        #x_dict = model.addVars(Time_periods, M, Patient_processes, Activities, vtype = GRB.BINARY, lb = 0.0, ub = GRB.INFINITY, name = "x")

        u_A_variable = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "u_A")
        u_B_variable = model.addVars(total_queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "u_B")

        model.update()

        #removing variables where n>m
        for j in range(total_queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        if n > m:
                            model.remove(c_variable[j, t, n, m])
                            model.remove(q_variable[j, t, n, m])

        #removing variables connected to queues not used in the current run
        for j in range(current_diagnosis_queues, total_queues - total_treatment_queues):
            for t in range(Time_periods):
                model.remove(b_variable[j, t])
                model.remove(u_A_variable[j, t])
                model.remove(u_B_variable[j, t])
                for n in range(N):
                    for m in range(M):
                        model.remove(c_variable[j, t, n, m])
                        model.remove(q_variable[j, t, n, m])

        end_variables = time.time()
        if not in_iteration == True:
            print("Generating variables:", end_variables - start_variables)
        #******************** Objective function ********************
        if not in_iteration:
            model.setObjective(gp.quicksum(mp.obj_weights(j, n) * q_variable[j, t, n, m]  for j in range(total_queues) for t in range(Time_periods) for n in range(N) for m in range(M)), GRB.MINIMIZE)
        else:
            model.setObjective(gp.quicksum(weights[j, n] * q_variable[j, t, n, m]  for j in range(total_queues) for t in range(Time_periods) for n in range(N) for m in range(M)), GRB.MINIMIZE)

        #******************** Constraints ********************

        start_constraints = time.time()
        #Updating queues whith serviced patients/patients
        for j in range(total_queues):
            if j < current_diagnosis_queues:
                for t in range(Time_periods):
                    for m in range(M):
                        if m == 0:
                            model.addConstr(q_variable[j, t, 0, 0] == Patient_arrivals_jt[j, t % 7] + gp.quicksum(G_jtnm[j, t, n, 0] for n in range(N)) + gp.quicksum(c_variable[i, t - M_j[i], 0, 0] * Q_ij[i, j] for i in range(current_diagnosis_queues) if (t - M_j[i]) >= 0))
                        elif t == 0 and m >= 1:
                            for n in range(N):
                                if n <= m:
                                    model.addConstr(q_variable[j, t, n, m] == E_jnm[j, n, m] + G_jtnm[j, t, n, m])
                        else:
                            model.addConstr(q_variable[j, t, 0, m] == gp.quicksum(G_jtnm[j, t, n, m] for n in range(N)) + gp.quicksum(c_variable[i, t - M_j[i], n, m] * Q_ij[i, j] for i in range(current_diagnosis_queues) for n in range(N) if (t - M_j[i]) >= 0))
            elif j >= total_queues - total_treatment_queues:
                for t in range(Time_periods):
                    for m in range(M):
                        if t == 0 and m >= 1:
                            for n in range(N):
                                model.addConstr(q_variable[j, t, n, m] == E_jnm[j, n, m] + G_jtnm[j, t, n, m])
                        elif mf.is_first_queue_in_treatment(j):
                            model.addConstr(q_variable[j, t, 0, m] == mp.share_of_patients_into_treatment * (1 / mf.get_number_of_treatment_paths()) * gp.quicksum(c_variable[i, t - M_j[i], n, m] for i in set_of_last_queues_in_diagnosis for n in range(N) if (t - M_j[i]) >= 0))
                        else:
                            model.addConstr(q_variable[j, t, 0, m] == gp.quicksum(G_jtnm[j, t, n, m] for n in range(N)) + gp.quicksum(c_variable[i, t - M_j[i], n, m] * Q_ij[i, j] for i in range(total_treatment_queues, total_queues) for n in range(N) if (t - M_j[i]) >= 0))
            else:
                continue

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
        for t in range(Time_periods):
            for r in range(Resources):
                model.addConstr(gp.quicksum(H_jr[j, r] * b_variable[j, t] for j in range(total_queues)) <=  L_rt[r, t % 7])


        """
        #TODO: endre settet A til et sett A hvor aktivitenene er "fase-sluttende" aktiviteter
        for t in range(Time_periods):
            for g in range(Patient_processes):
                for a in range(Activities):
                    j = mf.find_queue(g, a)
                    if j == -1:
                        continue
                    for m in range(M):
                        if m > Time_limits_ga[g, a]:
                            for n in range(N):
                                model.addConstr(q_variable[j, t, n, m] <= 0)
                                pass
        """

        #Shifting constraints
        for j in range(total_queues):
            for t in range(Time_periods - shift):
                model.addConstr(u_A_variable[j, t] - u_B_variable[j, t] == b_variable[j, t] - A_jt[j, t])


        #The number of shifts must be below a value K
        model.addConstr(gp.quicksum(u_A_variable[j, t] for j in range(total_queues) for t in range(Time_periods)) <= K)
        model.addConstr(gp.quicksum(u_B_variable[j, t] for j in range(total_queues) for t in range(Time_periods)) <= K)


        #Printing the time it took to generate the constraints
        end_constraints = time.time()
        if not in_iteration == True:
            print("Generating constraints:", end_constraints - start_constraints)
            print("\n")
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


    if not in_iteration == True:
        start_print = time.time()
        print("\n")
        print(colored("q(j, t, n, m)", 'green', attrs = ['underline']))
        mf.variable_printer("q", q_variable)
        print("\n")
        print(colored("c(j, t, n, m)", 'green', attrs = ['underline']))
        mf.variable_printer("c", c_variable)
        print("\n")
        print(colored("b(j, t)", 'green', attrs = ['underline']))
        mf.variable_printer("b", b_variable)
        #print("\n")
        #print(colored("x(t, m, g, a)", 'green', attrs = ['underline']))
        #mf.variable_printer("x", x_dict)
        print("\n")
        print(colored("u_A(j, t)", 'green', attrs = ['underline']))
        mf.variable_printer("u_A", u_A_variable)
        print("\n")
        print(colored("u_B(j, t)", 'green', attrs = ['underline']))
        mf.variable_printer("u_B", u_B_variable)

        model.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model.lp")
        model.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")

        number_of_constraints = len(model.getConstrs())
        number_of_variables = len(model.getVars())
        sum_exit_diagnosis, sum_exit_treatment = mf.number_of_exit_patients(c_variable, shift)
        mf.write_to_file(number_of_current_queues, Time_periods, N, M, Patient_processes, Activities, Resources, model.objVal, number_of_variables, number_of_constraints, runtime)

    #(husk at denne leser q-verdiene fra løsningen som akkurat er skrevet til fil)

        sum_D, sum_E, sum_G= mf.calculate_stats(current_diagnosis_queues, Patient_arrivals_jt, E_jnm, G_jtnm, shift)
        rollover_queue_next_period = np.sum(mf.create_E_jnm(total_queues, N, M, shift))
        c_jtnm_current = mf.old_solution("output/model_solution.sol", "c", 0)
        rollover_service_next_period = np.sum(mf.serviced_in_previous(total_queues, Time_periods, N, M, shift, c_jtnm_current, Q_ij, M_j))


        t = BeautifulTable()
        t.column_headers = [colored("Description", attrs = ['bold']), colored("Amount", attrs = ['bold'])]
        t.append_row(["Number of new patients entering the system", colored(sum_D*weeks, 'green')])
        t.append_row(["Number of rollover-queue patients entering the system", colored(sum_E, 'green')])
        t.append_row(["Number of rollover-service patients entering the system", colored(sum_G, 'green')])

        t.append_row(["Patients exiting from diagnosis", sum_exit_diagnosis])
        t.append_row(["Number of patients discharged after diagnosis", (1-mp.share_of_patients_into_treatment) * sum_exit_diagnosis])
        t.append_row(["Patients exiting from treatment", sum_exit_treatment])
        t.append_row(["Total patient exiting", colored((1-mp.share_of_patients_into_treatment) * sum_exit_diagnosis + sum_exit_treatment, 'red')])

        t.append_row(["Rollover-queue patients into next period", rollover_queue_next_period])
        t.append_row(["Rollover-service into next period", rollover_service_next_period])
        t.append_row(["Exiting patients / Incoming patients", ((1-mp.share_of_patients_into_treatment) * sum_exit_diagnosis + sum_exit_treatment) / (sum_D*weeks + sum_E + sum_G)])


        t.set_style(BeautifulTable.STYLE_BOX)
        print(t)

    #Outputing the time it took to print results and write to file
        end_print = time.time()
        print("Printing:", end_print - start_print)
        mg.create_gantt_chart(current_diagnosis_queues, Time_periods, mf.loadSolution("output/model_solution.sol")["b"])
    b_variable = mf.convert_dict(b_variable)
    q_variable = mf.convert_dict(q_variable)
    return q_variable, b_variable


#Running the model
def run_model():
    optimize_model(patient_processes = 1, weeks = 1, N_input = 15, M_input = 15, shift = 6, with_rolling_horizon = False, in_iteration = False, weights = 0)
    #shift = 0: Shifter ikke input-køene. Tar inn siste periode.


if __name__ == '__main__':
    run_model()
