import numpy as np



diagnosis_activity_dict={
0:"Referral",
1:"Gynecological investigation",
2:"Biopsi",
3:"CT",
4:"MRI",
5:"MDT",
6:"Outpatient clinic"}

treatment_activity_dict = {
0: "Chemoterapy",
1: "Surgery",
2: "Radiotherapy",
3: "Radiochemotherapy"
}

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

#Patient processes. Patient process on rows, activity on columns
#1 if patient process conducts activity, 0 otherwise
diagnostic_processes = np.array([
        [1,1,1,1,0,0,0],#Livmor. Start: 0
        [1,1,1,1,1,0,1], #Livmorhals. Start: 4
        [1,0,0,0,0,1,1]]) #Eggstokk. Start: 10

treatment_processes = np.array([
        [0,1,0,0],
        [0,0,0,1],
        [1,1,0,0]])

#recovery time after activity 0-5
diagnosis_recovery_times = np.array([1,1,1,1,1,1,1])

#recovery time after treatment activity 0-2
treatment_recovery_times = np.array([1,1,1,1])

#activities on rows, resources on columns. The entry represents the amount of resource j that is needed in activity i.
activity_resource_map = np.array([
        [1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0]])


#patient processes on rows, treatment paths on columns. Det entries denotes the
#probability of patients in in process i follows treatment path j. G x T
#Does not have to equal 1? Or last treatment is
probability_of_path = np.array([
        [1,0,0],
        [1/3,2/3,0],
        [0,0,1]])



#New demand in queue j in time period t
Patient_arrivals_jt=np.matrix([
        [10,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [10,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [10,0,0,0,0,0,0],#
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]])

#Max number of days allowed before activity a
Time_limits_j=np.array([100,6,100,16,100,6,100,100,100,16,100,6,16,100,100,100,100,100,100,100,100,100,100,100,100,100])

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
        [0,0,0,0,0,0,0,0,0,0],#
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],#
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],#
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],#
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],#
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
        [0,0,0,0,0,0,0,0,0,0]])



target_access_time = np.full((100, 100), 1)
target_service_performance = np.full((100, 100), 1)
