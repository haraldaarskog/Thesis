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
        return np.power(m, 1)
    else:
        return np.power(m, 1)

def is_queue_current(j, current_diagnosis_queues):
    if j < current_diagnosis_queues or (j >= mf.get_total_number_of_diagnosis_queues() and j < mf.get_total_number_of_queues()):
        return True
    return False


#beregner access time: finner den ventetiden n, hvor alpha-persentil av pasienter har ventet
def find_min_n(q_variable, j, t, N, M, alpha):
    arr1=[]
    for n_iter in range(N):
        val_1 = gp.quicksum(q_variable[j, t, n, m] for n in range(n_iter) for m in range(M) if n <= m)
        val_2 = gp.quicksum(q_variable[j, t, n, m] for n in range(N) for m in range(M) if n <= m)
        val_1 = val_1.getValue()
        val_2 = alpha * val_2.getValue()
        if val_1 > val_2:
            return n_iter
    return 0

#Regner ut access time for hver kombinasjon av j og t
def calculate_access_time(q_variable, J, T, N, M, alpha):
    access_time = np.zeros((J, T))
    for j in range(J):
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
            sum += b[key]

    return sum


def calculate_u(J, T, s, service_performance, epsilon, old_u):
    u_array = np.zeros(J)
    if s == 0:
        for j in range(J):
            u_array[j] = 1
    else:
        for j in range(J):
            #counts total number of services for queue j
            service_val = count_b_dict(service_performance, j, T - 1)
            print(service_val)
            if service_val == 0:
                value_1 = old_u[j] - 1/s
            else:
                target_service_val = np.sum(target_service_performance[j][:T-1])
                value_1 = old_u[j] + (1 / s) * (np.divide(target_service_val, service_val) - 1)
            if value_1 > epsilon:
                u_array[j] = value_1
            else:
                u_array[j] = epsilon
    return u_array




def calculate_m(J, T, s, access_time_performance, alpha, epsilon, old_m):
    m_array = np.zeros(J)
    if s == 0:
        for j in range(J):
            m_array[j] = 1 + 0.001
    else:
        for j in range(J):
            access_time_val = np.sum(access_time_performance[j][:T])
            target_access_time_val = np.sum(target_access_time[j][:T])
            value_1 = old_m[j] + (1 / s) * (np.divide(access_time_val, target_access_time_val) - 1)
            if value_1 > 1 + epsilon:
                m_array[j] = value_1
            else:
                m_array[j] = 1 + epsilon
    return m_array

def run_optimization(alpha, N, M, weeks, obj_weights, E, G):
    total_queues = mf.get_total_number_of_queues()
    E = None
    G = None
    K = 100
    q_variables, b_variables, _ = mm.optimize_model(weeks = weeks, N_input = N, M_input = M, shift = 6, with_rolling_horizon = False, in_iteration = True, weights = obj_weights, E = E, G = G, K = K)
    access_time = calculate_access_time(q_variables, total_queues, weeks * 7, N, M, alpha)
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

target_service_performance = np.array([
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1]
])

target_access_time = np.array([
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1],
[1,1,1,1,1,1,1]
])

def run_iterative_process(weeks, M, E, G, alpha, epsilon, theta):

    total_queues = mf.get_total_number_of_queues()
    s = 0
    time_periods = weeks * 7
    stopping_criteria = True
    previous_u_array = []
    previous_m_array = []
    service_performance = []
    access_time_performance = []
    N = int(np.round(M*3/5))

    while stopping_criteria:
        print("***********")
        print("Iteration:", s)
        print("***********")
        u_array = calculate_u(total_queues, time_periods, s, service_performance, epsilon, previous_u_array)
        m_array = calculate_m(total_queues, time_periods, s, access_time_performance, alpha, epsilon, previous_m_array)
        #print(u_array)
        #print(m_array)
        obj_weights = calculate_objective_weights(total_queues, N, u_array, m_array)
        #print(obj_weights)
        access_time_performance, service_performance = run_optimization(alpha, N, M, weeks, obj_weights, E, G)
        print("ATP")
        print(access_time_performance)
        print("SP")
        print(mf.from_dict_to_matrix(service_performance))
        stopping_criteria = convergence_check(s, total_queues, u_array, previous_u_array, m_array, previous_m_array, theta)
        previous_u_array = np.copy(u_array)
        previous_m_array = np.copy(m_array)
        s += 1
    #print(obj_weights)
    return obj_weights



if __name__ == '__main__':
    run_iterative_process(weeks =  1, M = 10, alpha = 0.5, epsilon = 0.001, theta = 0.1)
