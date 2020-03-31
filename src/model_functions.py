import numpy as np
import gurobipy as gp
from prettytable import PrettyTable
import pandas as pd
import model_parameters as mp
from datetime import datetime
import re

patient_processes=mp.patient_processes
activity_dict=mp.activity_dict

def number_of_queues(number_of_patient_processes):
    if number_of_patient_processes>patient_processes.shape[0]:
        raise ValueError("There are not defined that much patient processes.")
    sum=0
    for i in range(number_of_patient_processes):
        sum+=patient_processes[i][:].sum()
    return sum

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

def variable_printer(var_name,dict):

    for key in dict:

        value=dict[key]
        try:
            value=value.x
        except Exception as e:
            continue
        #if isinstance(value, gp.Var):

        if value>0:
            print('%s%s = %3.2f' % (var_name,key, value))

#returns dict. very nice
def loadSolution(file_name):
    set_of_variables=set()
    file = open(file_name, "r")
    res_dict={}
    d={}
    for line in file:
        if line[0]=="#":
            continue
        variable=line[0]
        arr=np.asarray(re.findall('\d+(?:\.\d+)?', line))
        value=arr[-1].astype(float)
        if variable not in set_of_variables:
            set_of_variables.add(variable)
            res_dict[variable]={}
        indices=arr[:-1].astype(int)
        res_dict[variable][tuple(indices)]=value
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

def create_empty_initial_queue(j,n,m):
    return np.zeros((j,n,m))

def create_normal_distribution(mean,std_deviation,n_samples):
    #mean, std_deviation = 100, 10 # mean and standard deviation
    s = np.random.normal(mean, std_deviation, n_samples)
    return np.round(s)


#Writing the r
def write_to_file(J,T,N,M,G,A,R,obj_value,num_days_last_service,n_variables,n_constraints,runtime):
    now = datetime.now()
    dt_string = now.strftime("%d/%m %H:%M:%S")
    date,time=dt_string.split(" ")
    runtime=np.round(runtime,4)
    df=pd.read_excel('run_data.xlsx', index_col=0)
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
            df.loc[index]=J,T,N,M,G,A,R,obj_value,num_days_last_service,n_variables,n_constraints,runtime,date,time
            flag=False
            break
    if flag==True:
        df.loc[len(df)]=[J,T,N,M,G,A,R,obj_value,num_days_last_service,n_variables,n_constraints,runtime,date,time]
    df = df.sort_values(by=['Date','Time'],ascending=False)
    df.to_excel('run_data.xlsx',index=True)

def serviced_in_previous(J,T,N,M,shift,c_dict):
    arr=np.zeros((J,T,N,M))
    for i in range(J):
        for j in range(J):
            for t in range(T):
                for n in range(N):
                    for m in range(M):
                        value=c_dict[i,t,n,m]
                        delay=mp.M_ij[i,j]
                        if value>0 and return_Q_ij(i,j)==1 and (t+delay)>=shift:
                            mod=(t+delay)%shift
                            arr[j,mod,n,m]=value
    return arr




def find_ga(queue,pp):
    rows,cols=pp.shape
    count=-1
    for r in range(rows):
        for c in range(cols):
            if pp[r,c]==1:
                count+=1
            if count==queue:
                return r,c


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

def return_Q_ij(i,j):
    pp=mp.patient_processes
    g_i,a_i=find_ga(i,pp)
    g_j,a_j=find_ga(j,pp)
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


def is_last_queue(j):
    sum1=-1
    for row in mp.patient_processes:
        sum1+=sum(row)
        if j==sum1:
            return True
    return False

def number_of_days_before_last_service(c_variable):
    sum_m=0
    sum_people=0
    for key in c_variable:
        value=c_variable[key]
        try:
            value=value.x
        except Exception as e:
            continue
        queue=key[0]
        if value>0 and is_last_queue(queue):
            sum_m+=key[-1]*value
            sum_people+=value
    return sum_m,sum_people

def calculate_stats(J,D_jt,E_jnm,G_jtnm):
    #antall initielt i kø
    sum_E=np.sum(E_jnm)
    #antall behandlede som kommer over en periode
    sum_G=np.sum(G_jtnm)
    #antall i kø som ikke er blitt behandlet enda ved tid T
    in_queue_last_time_period=np.sum(old_solution("output/model_solution.sol","q",0)[:,-1,:,:])
    #Antall innkommende pasienter
    sum_D=np.sum(D_jt[:J])
    return sum_D,sum_E,sum_G,in_queue_last_time_period
