import gurobipy as gp
import numpy as np
import model_functions as mf
import model_parameters as mp
import main_model as mm


def obj_weights(j, n):
    if mf.queue_is_treatment(j):
        return np.power(n,1)
    else:
        return np.power(n,1)

def obj_weights_m(j, m):
    if mf.queue_is_treatment(j):
        return np.power(m,1)
    else:
        return np.power(m,1)

def is_queue_current(j, current_diagnosis_queues):
    if j < current_diagnosis_queues or (j >= mf.get_total_number_of_diagnosis_queues() and j < mf.get_total_number_of_queues()):
        return True
    return False


def find_min_n(q_variable, j, t, N, M, alpha):
    arr1=[]
    for n_iter in range(N):
        val_1 = gp.quicksum(q_variable[j, t, n, m] for n in range(n_iter) for m in range(M) if n <= m)
        val_2 = gp.quicksum(q_variable[j, t, n, m] for n in range(N) for m in range(M) if n <= m)
        val_1 = val_1.getValue()
        val_2 = alpha * val_2.getValue()
        if val_1 > val_2:
            return n_iter - 1
    return 0


def calculate_access_time(q_variable, J, T, N, M, alpha, current_diagnosis_queues):
    access_time = np.zeros((J, T))
    for j in range(J):
        if is_queue_current(j, current_diagnosis_queues):
            for t in range(T):
                access_time[j, t] = find_min_n(q_variable, j, t, N, M, alpha)
    return access_time

def calculate_objective_weights(J, N, u, m):
    array = np.zeros((J, N))
    for j in range(J):
        for n in range(N):
            if n == 0:
                array[j, n] = 0
            else:
                array[j, n] = u[j] * np.power(m[j], n)
    return array

def count_b_dict(b, j, t):
    sum=0
    for key in b:
        j_b=key[0]
        t_b=key[1]
        if j_b == j and t_b <=t:
            sum+=b[key]

    return sum


def calculate_u(J, T, s, service_performance, epsilon, old_u, current_diagnosis_queues):
    u_array = np.zeros(J)
    if s == 0:
        for j in range(J):
            u_array[j] = 1
    else:
        for j in range(J):
            if is_queue_current(j, current_diagnosis_queues):
                service_val = count_b_dict(service_performance, j, T - 1)
                target_service_val = np.sum(mp.target_service_performance[j][:T-1])
                value_1 = old_u[j] + (1 / s) * (np.divide(target_service_val, service_val) - 1)
                if value_1 > epsilon:
                    u_array[j] = value_1
                else:
                    u_array[j] = epsilon
            else:
                continue
    return u_array



def calculate_m(J, T, s, access_time_performance, alpha, epsilon, old_m):
    m_array = np.zeros(J)
    if s == 0:
        for j in range(J):
            m_array[j] = 1 + 0.001
    else:
        for j in range(J):
            access_time_val = np.sum(access_time_performance[j][:T])
            target_access_time_val = np.sum(mp.target_access_time[j][:T])
            value_1 = old_m[j] + (1 / s) * (np.divide(access_time_val, target_access_time_val) - 1)
            if value_1 > 1 + epsilon:
                m_array[j] = value_1
            else:
                m_array[j] = 1 + epsilon
    return m_array

def run_optimization(alpha, J, weeks, N, M, current_diagnosis_queues, obj_weights):
    #q_variables, b_variables = mi.optimize_model(weeks, N, M, True, True, obj_weights, 6)
    q_variables, b_variables, _ = mm.optimize_model(1, weeks, N, M, 6, False, True, obj_weights)


    access_time = calculate_access_time(q_variables, J, weeks * 7, N, M, alpha, current_diagnosis_queues)
    return access_time, b_variables



def convergence_check(s, J, current_u, previous_u, current_m, previous_m, theta):
    if s == 0:
        return True
    for j in range(J):
        u_val = np.abs(current_u[j] - previous_u[j])
        m_val = np.abs(current_m[j] - previous_m[j])
        max_val = max(u_val, m_val)
        if max_val >= theta:
            return True
    return False



def run_iterative_process(J, weeks, N, M, alpha, epsilon, theta, current_diagnosis_queues):

    s = 0
    time_periods = weeks * 7
    stopping_criteria = True
    previous_u_array = []
    previous_m_array = []
    service_performance = []
    access_time_performance = []

    while stopping_criteria:
        u_array = calculate_u(J, time_periods, s, service_performance, epsilon, previous_u_array, current_diagnosis_queues)
        m_array = calculate_m(J, time_periods, s, access_time_performance, alpha, epsilon, previous_m_array)
        print(u_array)
        print(m_array)
        obj_weights = calculate_objective_weights(J, N, u_array, m_array)
        access_time_performance, service_performance = run_optimization(alpha, J, weeks, N, M, current_diagnosis_queues, obj_weights)
        stopping_criteria = convergence_check(s, J, u_array, previous_u_array, m_array, previous_m_array, theta)
        previous_u_array = np.copy(u_array)
        previous_m_array = np.copy(m_array)
        print("Iteration:", s)
        s += 1
    return obj_weights



if __name__ == '__main__':
    run_iterative_process(19, 1, 20, 20, 0.5, 0.001, 0.1, 15)
