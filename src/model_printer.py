from termcolor import colored, cprint
import model_functions as mf
import model_parameters as mp
from beautifultable import BeautifulTable
import sys
import numpy as np


def variable_printer(var_name, dict):
    for key in dict:
        value = dict[key]
        try:
            value = value.x
        except Exception as e:
            continue
        if value > 0.001:
            print('%s%s = %3.2f' % (var_name, key, value))


def print_variables(q_variable, c_variable, b_variable, u_A_variable, u_R_variable):
    print("\n")
    print(colored("q(j, t, n, m)", 'green', attrs = ['underline']))
    variable_printer("q", q_variable)
    print("\n")
    print(colored("c(j, t, n, m)", 'green', attrs = ['underline']))
    variable_printer("c", c_variable)
    print("\n")
    print(colored("b(j, t)", 'green', attrs = ['underline']))
    variable_printer("b", b_variable)
    print("\n")
    print(colored("u_A(j, t)", 'green', attrs = ['underline']))
    variable_printer("u_A", u_A_variable)
    print("\n")
    print(colored("u_R(j, t)", 'green', attrs = ['underline']))
    variable_printer("u_R", u_R_variable)



def print_patient_process(g):
    path = mp.diagnostic_processes[g][:]
    str=""
    for i in range(len(path)):
        str += mp.activity_dict[path[i]]+" -> "
    print(str[:-4])

def all_print_patient_processes(G):
    for g in range(G):
        print("DP: " + str(g) + ":",end=" ")
        print_patient_process(g)

def print_treatment_processes():
    tp = mp.treatment_processes
    rows = tp.shape[0]
    for r in range(rows):
        string = ""
        for i in range(len(tp[r])):
            string += mp.activity_dict[tp[r][i]] + " -> "
        print("TP: "+str(r)+": "+ string[:-4])



def create_set_table(number_of_current_queues, Time_periods, N, M, Patient_processes, Activities, Resources):
    t = BeautifulTable()
    t.column_headers=['Queues', 'Time periods', 'N', 'M', 'Patient processes', 'Activities', 'Resources']
    t.append_row([number_of_current_queues, Time_periods, N, M, Patient_processes, Activities, Resources])
    t.set_style(BeautifulTable.STYLE_BOX)
    print(t)

def print_model_status(status, runtime):
    if status == 2:
        print(colored("Found optimal solution in %g seconds (%g minutes)" % (runtime, (runtime / 60)), 'green', attrs = ['bold']))
    elif status == 3:
        print(colored("Model is infeasible", 'red', attrs = ['bold']))
        sys.exit()
    else:
        print(colored("Check gurobi status codes", 'red', attrs = ['bold']))
        sys.exit()

def create_overview_table(new_patient_arrivals, sum_E, sum_G):#, sum_exit_diagnosis, sum_exit_treatment, rollover_queue_next_period, rollover_service_next_period, discharged):
    t = BeautifulTable()
    t.column_headers = [colored("Description", attrs = ['bold']), colored("Amount", attrs = ['bold'])]
    t.append_row(["Number of new patients entering the system", colored(new_patient_arrivals, 'green')])
    t.append_row(["Number of rollover-queue patients entering the system", colored(sum_E, 'green')])
    t.append_row(["Number of rollover-service patients entering the system", colored(sum_G, 'green')])

    #t.append_row(["Patients exiting from diagnosis", sum_exit_diagnosis])
    #t.append_row(["Number of patients discharged after diagnosis", discharged])
    #t.append_row(["Patients exiting from treatment", sum_exit_treatment])
    #t.append_row(["Total patient exiting patient pathway", colored(discharged + sum_exit_treatment, 'red')])

    #t.append_row(["Rollover-queue patients into next period", colored(rollover_queue_next_period, 'red')])
    #t.append_row(["Rollover-service into next period", colored(rollover_service_next_period, 'red')])
    #t.append_row(["Exiting patients / Incoming patients", str(100 * np.around((discharged + sum_exit_treatment) / (new_patient_arrivals + sum_E + sum_G), decimals = 2))+"%"])

    t.set_style(BeautifulTable.STYLE_BOX)
    print(t)


def print_resource_utilization(J, T, b_jt):
    print("Printing resource utilization:")
    number_of_resources = mp.L_rt.shape[0]
    for r in range(number_of_resources):
        for t in range(T):
            sum = 0
            for j in range(J):
                try:
                    sum += b_jt[j, t] * mp.H_jr[j, r]
                except Exception as e:
                    continue
            if sum > 0:
                print("r("+str(r)+","+str(t)+") =",sum)
