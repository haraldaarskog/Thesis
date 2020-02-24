import numpy as np


J=2
T=D=2
N=10
R=3

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

def loadSolution(J,T,N, file_name):
    c_matrix=np.zeros((J,T,N))
    q_matrix=np.zeros((J,T,N))
    file = open(file_name, "r")
    for line in file:
        if line[0]=="#":
            continue
        line=line[0:-1]
        variable=line[0]
        queue=int(line[2])
        time_period=int(line[4])
        n_periods=int(line[6])
        variable_value=line[-1]
        if variable == "c":
            c_matrix[queue,time_period,n_periods]=variable_value
        elif variable == "q":
            q_matrix[queue,time_period,n_periods]=variable_value
        else:
            continue
    return c_matrix, q_matrix

def queue_to_pathway(queue_number,matrix):
    counter=-1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]==1:
                counter+=1
                if counter==queue_number:
                    return i
    raise ValueError("Something wrong happened")
    
