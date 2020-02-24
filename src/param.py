import numpy as np

#Objective value weight
W_jn=np.matrix([
        [1,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,8,9,10]])

#Fractions of patients moving from queue i to queue j
Q_ij=np.matrix([
        [0,1],
        [0,0]])

#New demand in queue j in time period t
D_jt=np.matrix([
        [2,2,2,2,2,2,2,2,2,2],
        [2,2,2,2,2,2,2,2,2,2]])

#Expected capacity requirements from resource type r for a patient in queue j
B_jr=np.matrix([
        [0,1],
        [0,0]])

#Number of patients already in queue when t=0
E_jn=np.matrix([
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0]])

#Expected delay from queue i to j
M_ij=np.matrix([
        [1,1],
        [1,1]])
