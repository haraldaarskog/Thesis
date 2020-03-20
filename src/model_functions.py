import numpy as np
import gurobipy as gp
from prettytable import PrettyTable
import pandas as pd
import model_parameters

patient_processes=model_parameters.patient_processes
activity_dict=model_parameters.activity_dict

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
        if isinstance(value, gp.Var):
            value=value.x
        if value>0:
            print('%s%s = %3.2f' % (var_name,key, value))

#returns 2dim dict. very nice
def loadSolution(file_name):
    set_of_variables=set()
    file = open(file_name, "r")
    res_dict={}
    d={}
    for line in file:
        if line[0]=="#":
            continue
        variable=line[0]
        value=int(line[-2])
        if variable not in set_of_variables:
            set_of_variables.add(variable)
            res_dict[variable]={}
        end=line.find(")")
        arr=[]
        for i in range(end):
            if line[i].isdigit():
                arr.append(int(line[i]))
        res_dict[variable][tuple(arr)]=value
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
