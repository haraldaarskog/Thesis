import numpy as np
import simulation as sim

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
        [0,1,2,3], #Livmor. 0/3
        [0,6,4,3,7], #Livmorhals. Start: 4/8
        [0,2,6,5]]) #Eggstokk. Start: 9/12

treatment_processes = np.array([
        [6,1,9], #13/17: Livmor, høyrisiko
        [6,1,9,8],#, #18/23: Livmor: tilleggsbehandling, 6 kurer cellegift
        [9], #24/26:Livmorhals
        [9,5,10], #27/31: Livmorhals
        [3,8,10,14,10,13,8,4], #32/39: Livmorhals
        [6,8,9,6,8], #40/46: Eggstokk
        [9,6,8]]) #47/51: Eggstokk


#patient processes on rows, treatment paths on columns. Det entries denotes the
#probability of patients in in process i follows treatment path j. G x T
#Does not have to equal 1? Or last treatment is
probability_of_path = np.array([
        [0.35,0.15,0,0,0,0,0],
        [0,0,0.27,0.03,0.6,0,0],
        [0,0,0,0,0,0.25,0.65]])

#recovery time after activity
activity_recovery_time = {
0: 0,
1: 0,
2: 0,
3: 0,
4: 0,
5: 0,
6: 0,
7: 3,
8: 7,
9: 4,
10: 2,
11: 0,
12: 1,
13: 2,
14: 0
}

activity_duration = {
0: 0,
1: 1,
2: 1,
3: 0.5,
4: 0.5,
5: 1,
6: 0.75,
7: 1,
8: 4,
9: 2,
10: 1,
11: 1,
12: 1,
13: 1,
14: 1
}


week_length = 7




#WEEKLY PATIENT DEMAND
uterine_demand = 4 * 1
cervical_demand = 3 * 1
ovarian_demand = 4 * 1


#New demand in queue j in time period t
ud = uterine_demand/5
cd = cervical_demand/5
od = ovarian_demand/5


Patient_arrivals_jt=np.matrix([
        [ud,ud,ud,ud,ud,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [cd,cd,cd,cd,cd,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [od,od,od,od,od,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]])


#Max Time Limit
m = 100
#First Time Limit
f = 60
#Second Time Limit
s = 220
#Third Time Limit
t = 360
#Max number of days allowed before activity a
Time_limits_j = np.array([
m,f,m,s,
m,f,m,m,s,
m,f,m,s,
t,m,m,
t,m,m,m,
t,
t,m,m,
t,m,m,m,m,m,m,m,
t,m,m,m,m,
t,m,m])


f1 = 6
s1 = 22
t1 = 36

Time_limits_test = np.array([
m,f1,m,s1,
m,f1,m,m,s1,
m,f1,m,s1,
t1,m,m,
t1,m,m,m,
t1,
t1,m,m,
t1,m,m,m,m,m,m,m,
t1,m,m,m,m,
t1,m,m])

"""
Time_limits_test = np.array([
m,f1,m,s1,
t1,m,m,
t1,m,m,m])
"""


#Resource capacity for resource r at day t in a week

L_rt = np.array([
        [45,45,45,45,45,0,0],#Physician0 - Begrensende
        [5,5,5,5,5,0,0],#Gynecologist1 - Begrensende
        [4,4,4,4,4,0,0],#Radiologist2
        [4,4,4,4,4,0,0],#Radiographer3
        [4,4,4,4,4,0,0],#Pathologist4 - Begrensende
        [0,0,0,0,0,0,0],#Surgeon5
        [45,45,45,45,45,0,0],#Nurse6 - begrensende
        [2,2,2,2,2,0,0],#CT scanner7
        [2,2,2,2,2,0,0], #MRI scanner8
        [16,16,16,16,16,0,0],#Operating room9
        [2,2,2,2,2,0,0], #laboratory(biopsy)10
        [4,4,4,4,4,0,0], #outpatient clinic11
        [0,0,0,0,0,0,0], #Bed, not used in model12
        [15,15,15,15,15,0,0], # Day unit13
        [3,3,3,3,3,0,0], # Radiotherapy laboratory14
        [0,0,0,0,0,0,0]]) # Meeting room15


resource_dict={
###Staff
0: "Physician",
1: "Gynecologist",
2: "Radiologist",
3: "Radiographer",
4: "Pathologist",
5: "Surgeon",
6: "Nurse",
##
7: "CT scanner",
8: "MRI scanner",
9: "Operating room",
10: "Laboratory (Biopsy)",
11: "Outpatient clinic",
12: "Bed",
13: "Day unit",
14: "Radiotherapy laboratory",
15: "Meeting room"}


activity_resource_dict = {
0: {},
1: {1:1, 6:2, 11:1},
2: {1:0.5, 4:0.5, 10:0.5},
3: {2:0.5, 3:1, 7:0.5},
4: {2:0.5, 3:1, 8:0.5},
5: {0:5},
6: {0:0.5, 6:0.5, 11:0.5},
7: {0:1, 1:1, 6:4, 9:1},
8: {0:4, 6:4, 13:2},
9: {0:3, 6:3, 9:3},
10: {2:1, 3: 1, 14: 1},
11: {},
12: {},
13: {2:1, 3:1, 14:1},
14: {0:0.5, 11:0.5}
}
