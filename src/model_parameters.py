import numpy as np
import model_functions as mf
activity_dict={
0:"Referral",
1:"Gynecological investigation",
2:"MDT",
3:"CT",
4:"MRI",
5:"Outpatient clinic"}

resource_dict={
0: "Physician",
1: "Gynecologist",
2: "Radiologist",
3: "Radiographer",
4: "Nurse",
5: "CT scanner",
6: "MRI scanner",
7: "Operating room",
8: "Surgeon",
9: "Laboratory"

}

share_of_patients_into_treatment = 1

#Patient processes. Patient process on rows, activity on columns
#1 if patient process conducts activity, 0 otherwise
patient_processes = np.array([
        [1,0,1,0,0,1],
        [1,1,0,0,1,1],
        [1,0,0,0,0,1],
        [1,1,0,0,0,1],
        [1,1,0,0,0,1]])

treatment_processes = np.array([
        [1,0,1],
        [1,0,0],
        [0,1,0]])

#recovery time after activity 0-5
diagnosis_recovery_times = np.array([1,1,1,1,1,1])

#recovery time after treatment 0-2
treatment_recovery_times = np.array([1,1,1])

#activities on rows, resources on columns. The entry represents the amount of resource j that is needed in activity i.
activity_resource_map=np.array([
        [1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]])


K=1000000

#Objective value weight
def obj_weights(j, n):
    if mf.queue_is_treatment(j):
        return np.power(n,1)
    else:
        return np.power(n,1)


#New demand in queue j in time period t
D_jt=np.matrix([
        [5,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [5,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [10,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [10,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [10,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]])

#Max number of days allowed before activity a
F_ga=np.array([
        [20,30,50,40,50,40,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100],
        [10,20,30,40,50,60,70,80,80,100]])

#Resource capacity for resource r at day t in a week
L_rt=np.array([
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8],
        [8,8,8,8,8,8,8]])

#Queue j's usage of resource r
H_jr=np.array([
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1]])
