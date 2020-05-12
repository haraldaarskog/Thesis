import numpy as np


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

activity_dict={
0: "Referral",
1: "Gynecological investigation",
2: "Biopsy",
3: "CT",
4: "MRI",
5: "MDT",
6: "Outpatient clinic",
7: "Day surgery with narcosis and MD decision",
8: "Chemoterapy",
9: "Surgery",
10: "Radiotherapy",
11: "Radiochemotherapy",
12: "Bed",
13: "Brachyterapy",
14: "Blood sample and meeting with physician"
}

#Patient processes. Patient process on rows, activity on columns
#1 if patient process conducts activity, 0 otherwise
diagnostic_processes = np.array([
        [0,1,2,3],#Livmor. Start: 0
        [0,6,4,3,7], #Livmorhals. Start: 4
        [0,2,6,5,6]]) #Livmorhals. Start: 9

treatment_processes = np.array([
        [6,1,12,9,12], #13 Livmor, høyrisiko
        [6,1,12,9,12,8], #Livmor: tilleggsbehandling, cellegidt
        [12,9,12], #Livmorhals
        [12,9,12,5,10], #Livmorhals
        [3,8,10,14,10,13,8,4], #Livmorhals
        [6,8,12,9,12,6,8], #Eggstokk
        [12,9,12,6,8]]) #Eggstokk


#patient processes on rows, treatment paths on columns. Det entries denotes the
#probability of patients in in process i follows treatment path j. G x T
#Does not have to equal 1? Or last treatment is
probability_of_path = np.array([
        [0.35,0.15,0,0,0,0,0],
        [0,0,7/24,1/24,2/3,0,0],
        [0,0,0,0,0,0.30,0.70]])

#recovery time after activity
diagnosis_recovery_times = np.full(100,1) #np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

#recovery time after treatment activity 0-2
treatment_recovery_times = np.full(100,1)#np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

#activities on rows, resources on columns. The entry represents the amount of resource j that is needed in activity i.
activity_resource_map = np.array([
        [1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]])






#New demand in queue j in time period t
Patient_arrivals_jt=np.matrix([
        [4,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [3,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [4,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]])

#Max number of days allowed before activity a
Time_limits_j = np.full(100,100) #np.array([100,6,100,22,100,6,100,100,100,22,100,6,22,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100])
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
H_jr = np.zeros((60,10))
"""
np.array([
        [0,0,0,0,0,0,0,0,0,0],#
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],#
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]])
"""


target_access_time = np.full((100, 100), 1)
target_service_performance = np.full((100, 100), 1)
