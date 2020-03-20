import numpy as np
import gurobipy as gp
from prettytable import PrettyTable
import pandas as pd

def createParameter(variables, J,T,N):
    matr=np.zeros((J,N))
    for v in variables:
        if v.x > 0:
            queue=int(v.varName[2])
            time_period=int(v.varName[4])
            n_periods=int(v.varName[6])
            if time_period==T-1:
                matr[queue,n_periods]=v.x
    return matr


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
    for key in d:
        arr1.append(key[0])#j
        arr2.append(key[1])#t
    return max(arr1), max(arr2)

def from_dict_to_matrix(d):
    j,t=find_max_of_dict(d)
    mat=np.zeros((j+1,t+1))
    for key in d:
        mat[key[0],key[1]]=d[key]
    return mat




def old_solution(file, shift):
    b_dict=loadSolution(file)["b"]
    b_dict_shifted=shift_solution(b_dict,shift)
    b_matrix=from_dict_to_matrix(b_dict_shifted)
    return b_matrix

#print(old_solution("output/model_solution.sol"))
