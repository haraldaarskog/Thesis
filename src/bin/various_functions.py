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
