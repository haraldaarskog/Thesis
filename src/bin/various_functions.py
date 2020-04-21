#bin
"""
#defining x_tmga
#TODO: Finn verdi for Big S
for g in range(Patient_processes):
    for a in range(Activities):
        j = mf.find_queue(g, a)
        if j == -1:
            continue
        for m in range(M):
            for t in range(Time_periods):
                model.addConstr(x_dict[t, m, g, a] <= gp.quicksum(c_dict[j, t, n, m] for n in range(N)))
                model.addConstr(10000 * x_dict[t, m, g, a] >= gp.quicksum(c_dict[j, t, n, m] for n in range(N)))

for t in range(Time_periods):
    for m in range(M):
        for g in range(Patient_processes):
            for a in range(Activities):
                #model.addConstr(m * x_dict[t, m, g, a] <= F_ga[g, a])
                pass

"""
"""
q(18, 6, 0, 3) = 0.33
q(18, 6, 1, 3) = 0.33
def return_Q_ij(i,j):
    pp=mp.patient_processes
    g_i,a_i=find_ga(i,pp)
    g_j,a_j=find_ga(j,pp)

    tp=mp.treatment_processes

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
"""
