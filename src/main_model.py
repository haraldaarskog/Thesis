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

start_initialization_time = time.time()
np.set_printoptions(suppress=True)

#A function that runs the model
#If with_rolling_horizon = True, input from the previous run is included
#If reset = True, input from previous run is deleted
#shift: number of days between running the model the last time and today. The number
#of days that are already implemented
def optimize_model(weeks, shift, with_rolling_horizon, reset):


    #********************Set sizes********************
    Patient_processes = 1

    #The number of queues is a function of the number of patient processes
    #and the number of activities in each patient process
    Queues = int(mf.number_of_queues(Patient_processes))
    Time_periods = 7 * weeks
    N = 15
    M = 15
    P = 3
    Activities = mp.patient_processes.shape[1]
    Resources = mp.activity_resource_map.shape[1]

    if shift >= Time_periods:
        print(colored("Shift must be less than the number of time periods", 'red', attrs = ['bold']))
        sys.exit()
    #********************Information print********************


    t = BeautifulTable()
    t.column_headers=['Queues', 'Time periods', 'N', 'M', 'Patient processes', 'Activities', 'Resources']
    t.append_row([Queues, Time_periods, N, M, Patient_processes, Activities, Resources])
    t.set_style(BeautifulTable.STYLE_BOX)
    print(t)

    mf.all_print_patient_processes(Patient_processes)


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
        E_jnm = mf.old_solution("output/model_solution.sol", "q", 0)[:, shift-1, :, :]
        c_jtnm_old = mf.old_solution("output/model_solution.sol", "c", 0)
        G_jtnm = mf.serviced_in_previous(Queues, Time_periods, N, M, shift, c_jtnm_old, Q_ij)
    else:
        E_jnm = np.zeros((Queues, N, M))
        A_jt = np.zeros((Queues, Time_periods))
        G_jtnm = np.zeros((Queues, Time_periods, N, M))


    #printing the initialization time
    end_initialization_time = time.time()
    print("Initialization took", end_initialization_time-start_initialization_time, "seconds")

    try:
        #******************** Model setup ********************

        model  =  gp.Model("mip_queue_model")
        model.setParam("OutputFlag", 0)

        #******************** Variables ********************
        total_diagnosis_queues=mf.get_total_number_of_diagnosis_queues()
        total_treatment_queues=mf.get_total_number_of_treatment_queues()
        start_variables = time.time()
        c_dict  =  model.addVars(Queues, Time_periods, N, M, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "c")
        q_dict  =  model.addVars(Queues, Time_periods, N, M, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "q")

        b_dict = model.addVars(Queues, Time_periods, vtype = GRB.INTEGER, lb = 0.0, ub = GRB.INFINITY, name = "b")
        w_dict = model.addVars(Queues, Time_periods, vtype = GRB.INTEGER, lb = 0.0, ub = GRB.INFINITY, name = "w")

        x_dict = model.addVars(Time_periods, M, Patient_processes, Activities, vtype = GRB.BINARY, lb = 0.0, ub = GRB.INFINITY, name = "x")

        u_A_dict = model.addVars(Queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "u_A")
        u_B_dict = model.addVars(Queues, Time_periods, vtype = GRB.CONTINUOUS, lb = 0.0, ub = GRB.INFINITY, name = "u_B")

        model.update()
        for j in range(Queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        if n>m:
                            model.remove(c_dict[j, t, n, m])
                            model.remove(q_dict[j, t, n, m])
        #model.update()
        end_variables = time.time()
        print("Generating variables:", end_variables - start_variables)
        #******************** Objective function ********************

        model.setObjective(gp.quicksum(mp.obj_weights(n) * q_dict[j, t, n, m]  for j in range(Queues) for t in range(Time_periods) for n in range(N) for m in range(M)), GRB.MINIMIZE)

        #******************** Constraints ********************

        start_constraints = time.time()

        #Updating queues whith serviced patients/patients. Denne restriksjonen tar lang tid å generere.
        for j in range(Queues):
            for t in range(Time_periods):
                for m in range(M):
                    if m == 0:
                        model.addConstr(q_dict[j, t, 0, 0] == D_jt[j, t % 7] + gp.quicksum(G_jtnm[j, t, n, 0] for n in range(N)) + gp.quicksum(c_dict[i, t - M_ij[i], 0, 0] * Q_ij[i, j] for i in range(Queues) if (t - M_ij[i]) >= 0))
                    elif t == 0 and m >= 1:
                        for n in range(N):
                            if n <= m:
                                model.addConstr(q_dict[j, t, n, m] == E_jnm[j, n, m] + G_jtnm[j, t, n, m])
                    else:
                        model.addConstr(q_dict[j, t, 0, m] == gp.quicksum(G_jtnm[j, t, n, m] for n in range(N)) + gp.quicksum(c_dict[i, t-M_ij[i], n, m] * Q_ij[i, j] for i in range(Queues) for n in range(N) if (t - M_ij[i]) >= 0))


        #Updating a queue when patients are serviced
        for j in range(Queues):
            for t in range(1, Time_periods):
                for n in range(1, N):
                    for m in range(1, M):
                        model.addConstr(q_dict[j, t, n, m] == q_dict[j, t - 1, n - 1, m - 1] - c_dict[j, t - 1, n - 1, m - 1])

        #Constriants ensuring that the number of serviced patients are equal to or less than then number of patients in the queue
        for j in range(Queues):
            for t in range(Time_periods):
                for n in range(N):
                    for m in range(M):
                        model.addConstr(c_dict[j, t, n, m] <= q_dict[j, t, n, m])

        #Defining b_jt
        for j in range(Queues):
            for t in range(Time_periods):
                model.addConstr(b_dict[j, t] == gp.quicksum(c_dict[j, t, n, m] for n in range(N) for m in range(M)))

        #Defining b_jt
        for j in range(Queues):
            for t in range(Time_periods):
                model.addConstr(w_dict[j, t] == gp.quicksum(q_dict[j, t, n, m] for n in range(N) for m in range(M)))


        #TODO: Finn verdi for Big S
        for g in range(Patient_processes):
            for a in range(Activities):
                j = mf.find_queue(g, a)
                if j == -1:
                    continue
                for m in range(M):
                    for t in range(Time_periods):
                        model.addConstr(x_dict[t, m, g, a] <= gp.quicksum(c_dict[j, t, n, m] for n in range(N)))
                        model.addConstr(10000 * x_dict[t, m, g, a] >= gp.quicksum(c_dict[j, t, n, m] for n in range(N)))

        #Resource constraints
        for t in range(Time_periods):
            for r in range(Resources):
                model.addConstr(gp.quicksum(H_jr[j, r] * b_dict[j, t] for j in range(Queues)) <=  L_rt[r, t % 7])


        for t in range(Time_periods):
            for m in range(M):
                for g in range(Patient_processes):
                    for a in range(Activities):
                        #model.addConstr(m * x_dict[t, m, g, a] <= F_ga[g, a])
                        pass

        #TODO: endre settet A til et sett A hvor aktivitenene er "fase-sluttende" aktiviteter
        for t in range(Time_periods):
            for g in range(Patient_processes):
                for a in range(Activities):
                    j = mf.find_queue(g, a)
                    if j == -1:
                        continue
                    for m in range(M):
                        if m > F_ga[g, a]:
                            for n in range(N):
                                model.addConstr(q_dict[j, t, n, m] <= 0)
                                pass

        #Shifting constraints
        for j in range(Queues):
            for t in range(Time_periods - shift):
                model.addConstr(u_A_dict[j, t] - u_B_dict[j, t] == b_dict[j, t] - A_jt[j, t])
                pass

        #The number of shifts must be below a value K
        model.addConstr(gp.quicksum(u_A_dict[j, t] for j in range(Queues) for t in range(Time_periods)) <= K)
        model.addConstr(gp.quicksum(u_B_dict[j, t] for j in range(Queues) for t in range(Time_periods)) <= K)


        #Printing the time it took to generate the constraints
        end_constraints = time.time()
        print("Generating constraints:", end_constraints - start_constraints)
        #******************** Optimize model ********************
        print("\n")
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
        print("\n")
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))



    start_print = time.time()
    print(colored("q(j, t, n, m)", 'green', attrs = ['underline']))
    mf.variable_printer("q", q_dict)
    print("\n")
    print(colored("c(j, t, n, m)", 'green', attrs = ['underline']))
    mf.variable_printer("c", c_dict)
    print("\n")
    print(colored("b(j, t)", 'green', attrs = ['underline']))
    mf.variable_printer("b", b_dict)
    print("\n")
    print(colored("x(t, m, g, a)", 'green', attrs = ['underline']))
    mf.variable_printer("x", x_dict)
    print("\n")
    print(colored("u_A(j, t)", 'green', attrs = ['underline']))
    mf.variable_printer("u_A", u_A_dict)
    print("\n")
    print(colored("u_B(j, t)", 'green', attrs = ['underline']))
    mf.variable_printer("u_B", u_B_dict)

    model.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model.lp")
    model.write("/Users/haraldaarskog/GoogleDrive/Masteroppgave/Git/Thesis/src/output/model_solution.sol")

    number_of_constraints = len(model.getConstrs())
    number_of_variables = len(model.getVars())
    num_days_last_service, exit_people = mf.number_of_days_before_last_service(c_dict)
    mf.write_to_file(Queues, Time_periods, N, M, Patient_processes, Activities, Resources, model.objVal, number_of_variables, number_of_constraints, runtime)

    #(husk at denne leser q-verdiene fra løsningen som akkurat er skrevet til fil)

    sum_D, sum_E, sum_G, in_queue_last_time_period = mf.calculate_stats(Queues, D_jt, E_jnm, G_jtnm, shift)



    t = BeautifulTable()
    t.column_headers = [colored("Description", attrs = ['bold']), colored("Amount", attrs = ['bold'])]
    t.append_row(["Patients exiting the system", exit_people])
    t.append_row(["Number of new patients entering the system", sum_D*weeks])
    t.append_row(["Number of rollover-queue patients entering the system", sum_E])
    t.append_row(["Number of rollover-service patients entering the system", sum_G])
    t.append_row(["Number of patients still in queue by last time period", in_queue_last_time_period])
    t.append_row(["Share of finished patients", exit_people / (sum_D*weeks + sum_E + sum_G)])
    t.set_style(BeautifulTable.STYLE_BOX)
    print(t)

    #Outputing the time it took to print results and write to file
    end_print = time.time()
    print("Printing:", end_print - start_print)
    mg.create_gantt_chart(Queues, Time_periods, mf.loadSolution("output/model_solution.sol")["b"])

#Running the model
def run_model():
    optimize_model(weeks = 2, shift = 7, with_rolling_horizon = True, reset = False)
    #shift = 0: Shifter ikke input-køene. Tar inn siste periode.

run_model()
