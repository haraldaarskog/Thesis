import numpy as np
import gurobipy as gp
import pandas as pd
import model_parameters as mp
from datetime import datetime
import re
import sys

patient_processes=mp.patient_processes
activity_dict=mp.activity_dict

output_file="output/model_solution.sol"

def number_of_queues(number_of_patient_processes):
    if number_of_patient_processes>patient_processes.shape[0]:
        raise ValueError("There are not defined that much patient processes.")
    sum=0
    for i in range(number_of_patient_processes):
        sum+=patient_processes[i][:].sum()
    return int(sum)

def find_queue(patient_pro,activity):
    if patient_processes[patient_pro, activity]==0:
        return -1
    rows,cols=patient_processes.shape
    sum=-1
    for p in range(rows):
        for a in range(cols):
            if patient_processes[p,a]==1:
                sum+=1
            if patient_pro==p and activity == a:
                return sum
    return None

def variable_printer(var_name, dict):

    for key in dict:

        value = dict[key]
        try:
            value = value.x
        except Exception as e:
            continue

        if value > 0.001:
            print('%s%s = %3.2f' % (var_name, key, value))

#returns dict. very nice
def loadSolution(file_name):
    set_of_variables = set()
    file = open(file_name, "r")
    res_dict = {}
    d = {}
    for line in file:
        if line[0] == "#":
            continue
        variable = line.split("[")[0]
        indices_1, value = line.split(" ")
        arr = np.asarray(re.findall('\d+(?:\.\d+)?', indices_1))
        value = float(value)
        if variable not in set_of_variables:
            set_of_variables.add(variable)
            res_dict[variable]={}
        indices_2=(np.round(arr.astype(float))).astype(int)
        res_dict[variable][tuple(indices_2)]=value
    return res_dict

def shift_solution(d, shift):
    d_new={}
    for key in d:
        j=key[0]
        old_t=key[1]
        if old_t>=shift:
            new_t=old_t-shift
            d_new[(j,new_t)] = d[key]
    return d_new


def find_max_of_dict(d):
    arr1=[]
    arr2=[]
    flag=True
    for key in d:

        if flag:
            key_len=len(key)
            arr=np.zeros(key_len)
            flag=False
        for i in range(key_len):
            value=int(key[i])
            if value>arr[i]:
                arr[i]=value
    return arr.astype(int)+1

def from_dict_to_matrix(d):
    dim=tuple(find_max_of_dict(d))
    array=np.zeros(dim)
    for key in d:
        array[key]=d[key]
    return array


def old_solution(file, variable, shift):
    d=loadSolution(file)[variable]
    if shift > 0:
        d=shift_solution(d,shift)
    array=from_dict_to_matrix(d)
    return array

def print_patient_process(g):
    path=patient_processes[g][:]
    str=""
    for i in range(len(path)):
        if path[i]==1:
            str+=activity_dict[i]+" -> "
    print(str[:-4])

def all_print_patient_processes(G):
    print("Patient processes:")
    for g in range(G):
        print(str(g)+":",end=" ")
        print_patient_process(g)


def update_queue(j,n,m, number_of_patients, queue):
    queue[j,n,m]=number_of_patients

def delete_all_queue_entries(queue):
    queue_dim=queue.shape
    return np.zeros(queue_dim)

def create_normal_distribution(mean,std_deviation,n_samples):
    #mean, std_deviation = 100, 10 # mean and standard deviation
    s = np.random.normal(mean, std_deviation, n_samples)
    return np.round(s)


#Writing the r
def write_to_file(J,T,N,M,G,A,R,obj_value,n_variables,n_constraints,runtime):
    now = datetime.now()
    dt_string = now.strftime("%d/%m %H:%M:%S")
    date,time=dt_string.split(" ")
    day,month=date.split("/")
    runtime=np.round(runtime,4)
    df=pd.read_excel('logging/run_data.xlsx', index_col=0)
    flag=True
    for index, row in df.iterrows():
        J_row=row[0]
        T_row=row[1]
        N_row=row[2]
        M_row=row[3]
        G_row=row[4]
        A_row=row[5]
        R_row=row[6]
        if J==J_row and T==T_row and N==N_row and M==M_row and G==G_row and A==A_row and R==R_row:
            df.loc[index]=J,T,N,M,G,A,R,obj_value,n_variables,n_constraints,runtime,day,month,time
            flag=False
            break
    if flag==True:
        df.loc[len(df)]=[J,T,N,M,G,A,R,obj_value,n_variables,n_constraints,runtime,day,month,time]
    df = df.sort_values(by=['Day',  'Month', 'Time'],ascending=[False, False, False])
    df.to_excel('logging/run_data.xlsx',index=True)



def calculate_rollover_serviced(J,T,N,M,c_variable, M_j,shift):
    sum=0
    for j in range(J):
        if is_last_queue_in_treatment(j):
            for t in range(T):
                for n in range(N):
                    for m in range(M):
                        try:
                            value = c_variable[j,t,n,m].x
                        except Exception as e:
                            continue
                        delay=M_j[j]
                        if value > 0 and (t + delay) > shift:
                            mod = (t + delay) % shift
                            sum += value
    return sum







def check_activity(a1,a2,g):
    count1=0
    count2=0
    if a1>=a2:
        return 0
    for i in range(len(g)):
        if g[i]==1:
            count1+=1
        if i==a1 and g[i]==1:
            count2=count1
        if i==a2 and count1==count2+1 and g[i]==1:
            return 1

    return 0

def is_last_queue_in_diagnosis(j):
    sum1=-1
    for row in mp.patient_processes:
        sum1+=sum(row)
        if j==sum1:
            return True
    return False


def number_of_exit_patients(c_variable, shift):
    sum_m = 0
    sum_exit_treatment = 0
    sum_exit_diagnosis = 0
    for key in c_variable:
        value = c_variable[key]
        try:
            value=value.x
        except Exception as e:
            continue
        queue=key[0]
        time=key[1]
        if value > 0 and is_last_queue_in_treatment(queue):
            sum_exit_treatment += value
        elif value > 0 and is_last_queue_in_diagnosis(queue):
            sum_exit_diagnosis += value
    return sum_exit_diagnosis, sum_exit_treatment

def calculate_stats(J,Patient_arrivals_jt,E_jnm,G_jtnm,shift):
    #antall initielt i kø
    sum_E=np.sum(E_jnm)
    #antall behandlede som kommer over en periode
    sum_G=np.sum(G_jtnm)
    #Antall innkommende pasienter
    sum_D=np.sum(Patient_arrivals_jt[:J])
    return sum_D,sum_E,sum_G


def get_min_capacity(j, T, R):
    min_val=100
    for r in range(R):
        for t in range(T):
            val=np.divide(mp.L_rt[r,t % 7], mp.H_jr[j,r])
            if val<min_val:
                min_val=val
    return min_val

def is_first_queue_in_treatment(j):
    sum1=np.sum(mp.patient_processes)
    for row in mp.treatment_processes:
        if j==sum1:
            return True
        sum1+=sum(row)
    return False

def get_total_number_of_diagnosis_queues():
    return np.sum(mp.patient_processes)

def get_total_number_of_treatment_queues():
    return np.sum(mp.treatment_processes)

def is_in_sequence_diagnosis(i, j):
    if i >= j:
        return False
    elif j >= i + 2:
        return False
    elif is_last_queue_in_diagnosis(i):
        return False
    else:
        return True


def is_last_queue_in_treatment(j):
    sum1 = -1 + get_total_number_of_diagnosis_queues()
    for row in mp.treatment_processes:
        sum1 += sum(row)
        if j == sum1:
            return True
    return False


#En dictioinary som returnerer lovlige stiretninger
def create_Q_ij():

    q_dict = {}
    d_queues = get_total_number_of_diagnosis_queues()
    t_queues = get_total_number_of_treatment_queues()
    Queues = d_queues + t_queues + 1

    for i in range(Queues):
        for j in range(Queues):
            if i >= j:
                q_dict[i, j] = 0
            elif j >= i + 2:
                q_dict[i, j] = 0
            elif is_last_queue_in_diagnosis(i):
                q_dict[i, j] = 0
            elif is_last_queue_in_treatment(i):
                q_dict[i, j] = 0
            else:
                q_dict[i, j] = 1
    return q_dict


def get_total_number_of_queues():
    return int(get_total_number_of_diagnosis_queues()+get_total_number_of_treatment_queues())

def queue_is_treatment(j):
    if j >= get_total_number_of_queues():
        print("Error, there are not that many queues defined")
        sys.exit()
    if j >= get_total_number_of_diagnosis_queues():
         return True
    return False


def find_ga(queue):
    if queue >= get_total_number_of_queues():
        print("Error!!!")
        sys.exit()
    pp = mp.patient_processes
    tp = mp.treatment_processes
    rows_pp,cols_pp = pp.shape
    rows_tp,cols_tp = tp.shape

    count = -1
    if not queue_is_treatment(queue):
        for r in range(rows_pp):
            for c in range(cols_pp):
                if pp[r, c] == 1:
                    count += 1
                if count == queue:
                    return r,c
    else:
        queue = queue % get_total_number_of_diagnosis_queues()
        for r_t in range(rows_tp):
            for c_t in range(cols_tp):
                if tp[r_t, c_t] == 1:
                    count += 1
                if count == queue:
                    return r_t, c_t


def create_M_j():
    m_dict={}
    Queues = get_total_number_of_queues()
    for i in range(Queues):
        g, a = find_ga(i)
        if queue_is_treatment(i):
            m_dict[i] = mp.treatment_recovery_times[a]
        else:
            m_dict[i] = mp.diagnosis_recovery_times[a]
    return m_dict


def generate_last_queues_in_diagnosis():
    queue_set=set()
    for j in range(get_total_number_of_diagnosis_queues()):
        if is_last_queue_in_diagnosis(j):
            queue_set.add(j)
    return queue_set


def get_number_of_treatment_paths():
    paths, activitites = mp.treatment_processes.shape
    return int(paths)

def serviced_in_previous(J,T,N,M,shift,c_variable,Q_ij, M_j):
    num=get_number_of_treatment_paths()
    arr=np.zeros((J,T,N,M))
    for i in range(J):
        for j in range(J):
            for t in range(shift + 1):
                for n in range(N):
                    for m in range(M):
                        value=c_variable[i,t,n,m]
                        delay=M_j[i]
                        if value > 0 and Q_ij[i,j]==1 and (t+delay)>shift:

                            mod=(t+delay)%shift
                            arr[j,mod,n,m]=value
                        elif value > 0 and is_last_queue_in_diagnosis(i) and is_first_queue_in_treatment(j) and (t+delay)>shift:

                            mod = (t+delay)%shift
                            arr[j,mod,n,m]=value/num
    return arr


def create_E_jnm(J, N, M, shift):
    E_jnm = old_solution("output/model_solution.sol", "q", 0)[:, shift, :, :]
    old_c = old_solution("output/model_solution.sol", "c", 0)[:, shift, :, :]
    new_array=np.zeros((J,N,M))
    for j in range(J):
        for n in range(N):
            for m in range(M):
                value_q = E_jnm[j,n,m]
                value_c = old_c[j,n,m]
                if value_q > 0.0001:
                    new_array[j, n + 1, m + 1] = value_q - value_c
    return new_array



def convert_dict(d):
    new_d={}
    for key in d:
        try:
            new_d[key] = d[key].x
        except Exception as e:
            continue
    return new_d

#OLD functions, can be used for later
"""
q(18, 6, 0, 3) = 0.33
q(18, 6, 1, 3) = 0.33
def return_Q_ij(i,j):
    pp=mp.patient_processes
    g_i,a_i=find_ga(i,pp)
    g_j,a_j=find_ga(j,pp)

    tp=mp.treatment_processes

    if pp[g_i,a_i]==0 or pp[g_j,a_j]==0:
        return 0
    elif g_i!=g_j:
        return 0
    elif a_i>=a_j:
        return 0
    elif check_activity(a_i,a_j,pp[g_i]):
        return 1
    else:
        return 0
"""
