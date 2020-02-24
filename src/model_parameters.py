import numpy as np

set_of_activities={"Referral","Gynecological investigation","MDT"}

#Patient processes. Patient process on rows, activity on columns
patient_processes=np.matrix([
        [1,0,1],
        [1,1,1]])

def number_of_queues(number_of_patient_processes):
    if number_of_patient_processes>patient_processes.shape[0]:
        raise ValueError("There are not defined that much patient processes.")
    sum=0
    for i in range(number_of_patient_processes):
        sum+=patient_processes[i][:].sum()
    return sum

#Objective value weight
W_jn=np.matrix([
        [1,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,8,9,10]])

#Fractions of patients moving from queue i to queue j
Q_ij=np.matrix([
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1],
        [0,0,0,0,0]])

#New demand in queue j in time period t
D_jt=np.matrix([
        [1,2,2,2,2,2,2,2,2,2],
        [1,2,2,2,2,2,2,2,2,2],
        [1,2,2,2,2,2,2,2,2,2],
        [1,2,2,2,2,2,2,2,2,2],
        [1,2,2,2,2,2,2,2,2,2]])

#Expected capacity requirements from resource type r for a patient in queue j
B_jr=np.matrix([
        [0,1],
        [0,0]])

#Number of patients already in queue when t=0
E_jn=np.matrix([
        [0,3,0,0,0,0,0,0,0,0,0],
        [0,3,0,0,0,0,0,0,0,0,0],
        [0,3,0,0,0,0,0,0,0,0,0],
        [0,3,0,0,0,0,0,0,0,0,0],
        [0,3,0,0,0,0,0,0,0,0,0]])

#Expected delay from queue i to j
M_ij=np.matrix([
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1]])

    
    
    
    
