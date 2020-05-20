import numpy as np
import matplotlib.pyplot as plt
import time
import main_model as mm
import model_functions as mf
import model_parameters as mp
import sys



#TODO: LES LEFTINK

#Master class
class Simulation:
    def __init__(self, all_queues, appointments, no_show_prob):
        self.all_queues = all_queues
        self.day = -1
        self.time = 0
        self.total_num_in_queue = [0]
        self.queue_development = {}
        for queue in self.all_queues:
            self.queue_development[queue.id] = [0]

        self.num_arrivals_in_system = [0]
        self.num_departures_in_system = [0]

        self.total_wait_array = []

        self.day_array=[0]

        self.appointments = appointments
        self.no_show_prob = no_show_prob
        self.distribute_appointments()
        self.total_patients_generated = 0
        self.total_patients_exited = 0
        self.patient_exit_list = []


    def create_G_matrix(self):
        g_dict = {}
        for queue in self.all_queues:
            for patient in queue.incoming_patients:
                j = queue.id
                t = patient.next_queue_arrival_day % 7
                m = patient.number_of_days_in_system
                if (j, t, m) not in g_dict.keys():
                    g_dict[j,t,m] = 1
                else:
                    g_dict[j,t,m] = g[j,t,m] + 1
        return g_dict

    def create_E_matrix(self):
        e_dict = {}
        N = M = 100
        for queue in self.all_queues:
            patient_array = queue.patient_list + queue.no_show_list
            queue_id = queue.id
            for patient in patient_array:
                for n in range(0, N):
                    for m in range(0, M):
                        if patient.number_of_days_in_queue == n and patient.number_of_days_in_system == m:
                            if (queue_id,n,m) not in e_dict.keys():
                                e_dict[queue_id,n,m] = 1
                            else:
                                e_dict[queue_id,n,m] = e_dict[queue_id,n,m] + 1
        return e_dict


    def distribute_appointments(self):
        for j in range(self.appointments.shape[0]):
            for q in self.all_queues:
                q.prob_of_no_show = self.no_show_prob
                if j == q.id and q is not None:
                    j_appointments = self.appointments[j]
                    q.appointments = j_appointments
                    print(q.id, q.appointments)

    def update_appointments(self, scheduled_appointments):
        for j in range(scheduled_appointments.shape[0]):
            for q in self.all_queues:
                if j == q.id and q is not None:
                    j_appointments = scheduled_appointments[j]
                    q.set_appointments(j_appointments)
                    print(q.id, q.appointments)


    def update_queue_development(self):
        self.day_array.append(self.day)
        total_num = 0
        sum_departs = 0
        sum_arrivals = 0
        for queue in self.all_queues:
            arr = self.queue_development[queue.id]
            arr.append(queue.get_number_of_patients_in_patientlist_noshowlist())
            self.queue_development[queue.id] = arr
            sum_arrivals += queue.num_arrivals
            sum_departs += queue.num_departs
            total_num += queue.get_number_of_patients_in_patientlist_noshowlist()
        self.total_num_in_queue.append(total_num)
        self.num_arrivals_in_system.append(sum_arrivals)
        self.num_departures_in_system.append(sum_departs)

    def update_queue_days_and_capacities(self):
        for queue in self.all_queues:
            queue.day = self.day
            queue.update_appointment_capacity()
            for p in queue.no_show_list:
                queue.no_show_list.remove(p)
                queue.patient_list.append(p)
            for patient in queue.patient_list:
                patient.new_day()
                patient.no_show = False


    def load_patients_in_service_into_next_queue(self):
        print("Patients from previous queues:")
        for queue in self.all_queues:
            queue.load_incoming_patients()


    def update_departure_times(self):
        for queue in self.all_queues:
            if queue.get_number_of_patients_in_queue() > 0 and queue.get_appointment_capacity() > 0:
                queue.set_next_departure_time(0)
            else:
                queue.set_next_departure_time(float('inf'))


    def get_queue(self, queue_id):
        for q in self.all_queues:
            if q.id == queue_id:
                return q
        return None

    def next_day(self):
        self.day += 1
        print("\nDAY", self.day)
        self.update_queue_days_and_capacities()
        self.generate_todays_arrivals()
        self.load_patients_in_service_into_next_queue()

        self.update_departure_times()

        self.time = 0
        while self.remaining_events_this_day():
            self.advance_time()
        self.update_queue_development()


    def generate_todays_arrivals(self):
        print("Todays arrivals:")
        sum = 0
        for queue in self.all_queues:
            if queue.is_incoming_queue:
                new_arrivals = queue.generate_arrivals_from_poisson()
                sum += new_arrivals
        self.total_patients_generated += sum

    def remaining_events_this_day(self):
        arr_queue, arr_patient, min_t = self.find_next_arrival()
        dep = self.find_next_departure()
        if arr_queue != None or dep != None:
            return True
        return False

    def find_next_departure(self):
        return_queue = None
        min_t = float('inf')
        for queue in self.all_queues:
            if queue.get_next_departure_time() < min_t and queue.get_appointment_capacity() > 0 and queue.get_number_of_patients_in_queue() > 0:
                min_t = queue.get_next_departure_time()
                return_queue = queue

        return return_queue

        #returning the queue with the lowest time today
    def find_next_arrival(self):
        return_queue = None
        return_patient = None
        min_t = float('inf')
        for queue in self.all_queues:
            if queue.get_next_arrival() is not None:
                time, patient = queue.get_next_arrival()
                if time is not None:
                    if min_t > time:
                        min_t = time
                        return_patient = patient
                        return_queue = queue
        return return_queue, return_patient, min_t


    def advance_time(self):
        #Next arrival
        if self.find_next_arrival() is not None:
            next_arrival_queue, arrival_patient, next_arrival_time = self.find_next_arrival()
        else:
            next_arrival_time = float('inf')

        #Next departure
        next_departure_queue = self.find_next_departure()

        if next_departure_queue == None:
            next_departure_time = float('inf')
        else:
            next_departure_time = next_departure_queue.get_next_departure_time()


        if next_arrival_time == float('inf') and next_departure_time == float('inf'):
            print("Ingenting skjer nå")
            return

        t_event = min(next_arrival_time, next_departure_time)
        self.time = t_event

        print("\nTIME IS ADVANCED (t = " + str(self.time) +")")
        print("Arrival queue:",next_arrival_queue,"at t =", next_arrival_time)
        print("Departure queue", next_departure_queue,"at t =", next_departure_time)

        if next_arrival_time <= next_departure_time:
            next_arrival_queue.handle_arrival_event(arrival_patient, self.time)
            next_arrival_queue.remove_patient_from_incoming_list(arrival_patient)

        else:
            no_show = next_departure_queue.handle_depart_event(self.time)
            if next_departure_queue.is_last_queue and not no_show and next_departure_queue.is_treatment_queue:
                self.total_patients_exited += 1
                print("MINUS______________")



class Queue:
    def __init__(self, id, next_Queue, is_incoming_queue, arrival_rate, is_treatment_queue):
        self.id = id
        self.day = 0
        self.is_treatment_queue = is_treatment_queue

        self.patient_list = []
        self.no_show_list = []
        self.prob_of_no_show = 0
        self.num_in_queue_development = []
        self.is_incoming_queue = is_incoming_queue
        self.is_last_queue = True if next_Queue == None else False

        self.next_departure_time = float('inf')

        self.num_arrivals = 0
        self.num_departs = 0
        self.next_Queue = next_Queue
        self.num_recovery_days = 1
        self.appointments = []
        self.appointment_capacity = 0
        self.appointment_dict = {}

        self.expected_number_of_arrivals_per_day = arrival_rate

        #Patients that is transfered between queues
        self.incoming_patients = []

        self.allowed_time = 24

        self.queue_graph = []

        self.potential_treatment_queues = []
        self.probability_of_treatment_queues = []

    def __str__(self):
     return "Queue ("+str(self.id)+"): " + str(self.get_number_of_patients_in_queue())

    #SETTERS
    def set_next_departure_time(self, time):
        self.next_departure_time = time

    def set_appointments(self, appointment_array):
        #print("1:",self.appointments)
        #print("2:",appointment_array)
        self.appointments = [*self.appointments, *appointment_array]
        #print("3:",self.appointments)



    def set_recovery_days(self, rec_days):
        self.num_recovery_days = rec_days

    #GETTERS


    def get_number_of_patients_in_queue(self):
        return len(self.patient_list)

    def get_number_of_patients_in_patientlist_noshowlist(self):
        return len(self.patient_list) + len(self.no_show_list)


        #finds the next arrival time of today
    def get_next_arrival(self):
        if len(self.incoming_patients) == 0:
            return None
        min_t = float('inf')
        next_patient = None
        for patient in self.incoming_patients:
            if patient.next_queue_arrival_day == self.day:
                if patient.next_queue_arrival_time < min_t:
                    min_t = patient.next_queue_arrival_time
                    next_patient = patient
        return min_t, next_patient


    def get_appointment_capacity(self):
        return self.appointment_capacity

    def get_next_departure_time(self):
        return self.next_departure_time

    #REMOVE
    def remove_patient_from_incoming_list(self, patient):
        patient_id = patient.id
        for p in self.incoming_patients:
            if p.id == patient_id:
                self.incoming_patients.remove(p)



    #GENERATORS
    def generate_service_time(self):
        mean_service_time = 1

        mean = 0
        std_deviation = 0.25
        service_uncertainty = np.random.normal(mean, std_deviation, 1)[0]
        if service_uncertainty + mean_service_time < 0:
            return mean_service_time
        else:
            return mean_service_time + service_uncertainty

    def generate_arrivals_from_poisson(self):
        new_arrivals = np.random.poisson(self.expected_number_of_arrivals_per_day, 1)[0]
        print(new_arrivals,"new arrivals in queue",self.id)
        for i in range(new_arrivals):
            self.handle_arrival_event(Patient(self.day), 0)
        return new_arrivals

    #OTHER

    def load_incoming_patients(self):
        for patient in self.incoming_patients:
            if patient.next_queue_arrival_day == self.day:
                self.handle_arrival_event(patient, 0)
                self.remove_patient_from_incoming_list(patient)

    def update_appointment_capacity(self):
        self.appointment_capacity = self.appointments[self.day]
        print("Capacity of queue:",self.id,"is:",self.appointment_capacity)

    def decrease_appointment_capacity(self):
        if self.appointment_capacity > 0:
            self.appointment_capacity -= 1
        else:
            print("Kapasiteten er allerede 0")


    #Patient methods
    def add_patient(self, patient, time):
        if patient is None:
            print("patient is None")
            sys.exit()
        self.patient_list.append(patient)
        patient.is_added_to_a_queue(time, self.day, self.id)
        self.num_arrivals += 1
        self.queue_graph.append([self.day + time/24, self.get_number_of_patients_in_queue()])

    def get_patient_with_highest_m(self):
        if len(self.patient_list) == 0:
            print("what")
            sys.exit()
        max_value = 0
        return_patient = None
        for p in self.patient_list:
            if p.get_number_of_days_in_system() > max_value:
                max_value = p.get_number_of_days_in_system()
                return_patient = p
        if return_patient is None:
            return_patient = self.get_FIFO_patient()
        return return_patient


    def remove_patient(self, patient):
        patient_id = patient.id
        for p in self.patient_list:
            if p.id == patient_id:
                self.patient_list.remove(p)
                break


    def get_FIFO_patient(self):
        if len(self.patient_list) == 0:
            print("Heeeeee")
            sys.quit()
            return None
        min_arrival_day = float('inf')
        min_arrival_time = float('inf')
        return_patient = None
        for patient in self.patient_list:
            if patient.get_queue_arrival_day() < min_arrival_day:
                if patient.get_queue_arrival_time() < min_arrival_time:
                    min_arrival_day = patient.get_queue_arrival_day()
                    min_arrival_time = patient.get_queue_arrival_time()
                    return_patient = patient
        return return_patient



    #HANDLE ARRIVALS AND DEPARTURES
    def handle_arrival_event(self, patient, time):
        self.add_patient(patient, time)
        print("EVENT: Arr: q("+str(self.id)+"), in queue: " + str(self.get_number_of_patients_in_queue())+", t =", time)
        if self.get_number_of_patients_in_queue() <= 1 and self.get_appointment_capacity() > 0:
            self.set_next_departure_time(time)


    def no_show_is_happening(self):
        prob = self.prob_of_no_show
        no_show = np.random.choice([0, 1], p=[1-prob, prob])
        if no_show == 1:
            return True
        else:
            return False

    #A departure means that a patients leaves the queue and is going into service
    def handle_depart_event(self, time):
        if len(self.patient_list) == 0:
            print("Queue",self.id,"is empty")
            sys.exit()

        departure_patient = self.get_patient_with_highest_m()
        self.decrease_appointment_capacity()
        if self.no_show_is_happening():
            print("We have a no show!")
            self.remove_patient(departure_patient)
            self.no_show_list.append(departure_patient)
            return True
        self.remove_patient(departure_patient)
        departure_patient.is_removed_from_queue()
        self.queue_graph.append([self.day + time/24, self.get_number_of_patients_in_queue()])
        self.num_departs += 1

        print("EVENT: Dep: q("+str(self.id)+"), remaining: "+ str(self.get_number_of_patients_in_queue()) + ", t =", time)
        service_time_for_departure_patient = time + self.generate_service_time()

        if self.get_number_of_patients_in_queue() > 0 and self.get_appointment_capacity() > 0:
            #Hvis det er mer tid tilgjengelig, gjennomfør neste time samme dag
            if service_time_for_departure_patient < self.allowed_time:
                self.set_next_departure_time(service_time_for_departure_patient)
            #Hvis ikke mer tid, sett tid til inf
            else:
                self.set_next_departure_time(float('inf'))

        else:
            self.set_next_departure_time(float('inf'))

        if not self.is_last_queue:
            self.transfer_patient_to_next_queue(departure_patient, service_time_for_departure_patient)
        elif self.is_last_queue and not self.is_treatment_queue:
            self.transfer_patient_to_treatment_path(departure_patient, service_time_for_departure_patient)

        return False


    def choose_treatment_path(self):
        index = np.random.choice(np.arange(0, len(self.potential_treatment_queues)+1), p=self.probability_of_treatment_queues)
        #Out of the system
        if index == len(self.potential_treatment_queues):
            print("OUT!!!!!!!!")
            return False
        return self.potential_treatment_queues[index]

    def transfer_patient_to_treatment_path(self, patient, service_time):
        next_treatment_queue = self.choose_treatment_path()
        if next_treatment_queue != False:
            patient.next_queue_arrival_day = self.day + self.num_recovery_days
            if self.num_recovery_days > 0:
                patient.next_queue_arrival_time = 0
            else:
                patient.next_queue_arrival_time = service_time

            next_treatment_queue.incoming_patients.append(patient)


    #Only patients that are moving forward to another queue
    def transfer_patient_to_next_queue(self, patient, service_time):

        patient.next_queue_arrival_day = self.day + self.num_recovery_days
        if self.num_recovery_days > 0:
            patient.next_queue_arrival_time = 0
        else:
            patient.next_queue_arrival_time = service_time

        self.next_Queue.incoming_patients.append(patient)









class Patient:


    def __init__(self, day):
        self.day = day
        self.entering_day = day

        self.number_of_days_in_queue = 0
        self.number_of_days_in_system = 0

        self.arrival_time_in_current_queue = float('inf')
        self.arrival_day_in_current_queue = float('inf')
        self.departure_time_from_current_queue = float('inf')

        self.next_queue_arrival_day = float('inf')
        self.next_queue_arrival_time = float('inf')
        self.queue_history = []

        self.id = id(self)

        self.is_in_queue = None


    def get_number_of_days_in_system(self):
        return self.number_of_days_in_system

    def update_queue_history(self, queue_id):
        self.queue_history.append(queue_id)

    def is_removed_from_queue(self):
        self.number_of_days_in_queue = 0
        self.is_in_queue = False

    def is_added_to_a_queue(self, time, day, queue_id):
        self.set_arrival_day_in_current_queue(day)
        self.set_arrival_time_in_current_queue(time)
        self.update_queue_history(queue_id)
        self.is_in_queue = True

    def new_day(self):
        self.day += 1
        self.number_of_days_in_queue += 1
        self.number_of_days_in_system += 1

    def get_queue_arrival_day(self):
        return self.arrival_day_in_current_queue

    def get_queue_arrival_time(self):
        return self.arrival_time_in_current_queue

    def set_arrival_day_in_current_queue(self, day):
        self.arrival_day_in_current_queue = day

    def set_arrival_time_in_current_queue(self, time):
        self.arrival_time_in_current_queue = time






def create_graph_1(s):
    for key in s.queue_development:
        dev = s.queue_development[key]
        plt.plot(s.day_array, dev, linestyle='-', label="Queue " + str(key))

    plt.xlabel('Days')
    plt.ylabel('Number of patients in queue')
    plt.legend(loc='best')
    plt.title('Simulation')
    plt.grid(True)
    plt.savefig('simulation/sim_figures/simulation_1.png')
    plt.close()

def create_graph_2(s):
    for queue in s.all_queues:
        dev = queue.queue_graph
        dev = np.asarray(dev)
        x_ax = dev[:,0]
        y_ax = dev[:,1]
        plt.plot(x_ax, y_ax,linestyle='-', label="Queue " + str(queue.id))

    plt.xlabel('Days')
    plt.ylabel('Number of patients in queue')
    plt.legend(loc='best')
    plt.title('Simulation')
    plt.grid(True)
    plt.savefig('simulation/sim_figures/simulation_2.png')

def create_graph_3(s, diagnosis):
    uterin_cancer=[0,1,2,3,14,15,16,17,18,19,20,21,22,23,24]
    cerivcal_cancer = [4,5,6,7,8,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    ovarian_cancer=[9,10,11,12,13,41,42,43,44,45,46,47,48,49,50,51,52]

    for queue in s.all_queues:
        if queue.id in uterin_cancer and diagnosis == "uterin":
            dev = s.queue_development[queue.id]
            plt.plot(s.day_array, dev,linestyle='-', label="Queue " + str(queue.id))


        if queue.id in cerivcal_cancer and diagnosis == "cervical":
            dev = s.queue_development[queue.id]
            plt.plot(s.day_array, dev,linestyle='-', label="Queue " + str(queue.id))


        if queue.id in ovarian_cancer and diagnosis == "ovarian":
            dev = s.queue_development[queue.id]
            plt.plot(s.day_array, dev,linestyle='-', label="Queue " + str(queue.id))

    plt.xlabel('Days')
    plt.ylabel('Number of patients in queue')
    plt.legend(loc='best')
    plt.title('Simulation')
    plt.grid(True)
    plt.savefig("simulation/sim_figures/simulation_"+diagnosis+".png")
    plt.close()




def main():
    np.random.seed(0)
    start_time=time.time()

    #Livmor
    q3 = Queue(3, None, False, None, False)
    q2 = Queue(2, q3, False, None, False)
    q1 = Queue(1, q2, False, None, False)
    q0 = Queue(0, q1, True, 4/7, False)

    #Livmor_treat 1
    q8 = Queue(8, None, False, None, True)
    q7 = Queue(7, q8, False, None, True)
    q6 = Queue(6, q7, False, None, True)
    q5 = Queue(5, q6, False, None, True)
    q4 = Queue(4, q5, False, None, True)

    #Livmor_treat 2
    q14 = Queue(14, None, False, None, True)
    q13 = Queue(13, q14, False, None, True)
    q12 = Queue(12, q13, False, None, True)
    q11 = Queue(11, q12, False, None, True)
    q10 = Queue(10, q11, False, None, True)
    q9 = Queue(9, q10, False, None, True)
    """

    #Livmorhals
    q8 = Queue(8, None, False, None, False)
    q7 = Queue(7, q8, False, None, False)
    q6 = Queue(6, q7, False, None, False)
    q5 = Queue(5, q6, False, None, False)
    q4 = Queue(4, q5, True, 3/7, False)

    #Eggstokk
    q13 = Queue(13, None, False, None, False)
    q12 = Queue(12, q13, False, None, False)
    q11 = Queue(11, q12, False, None, False)
    q10 = Queue(10, q11, False, None, False)
    q9 = Queue(9, q10, True, 4/7, False)


    #Treatment path: Livmor 1
    q18 = Queue(18, None, False, None, True)
    q17 = Queue(17, q18, False, None, True)
    q16 = Queue(16, q17, False, None, True)
    q15 = Queue(15, q16, False, None, True)
    q14 = Queue(14, q15, False, None, True)

    #Treatment path: Livmor 2
    q24 = Queue(24, None, False, None, True)
    q23 = Queue(23, q24, False, None, True)
    q22 = Queue(22, q23, False, None, True)
    q21 = Queue(21, q22, False, None, True)
    q20 = Queue(20, q21, False, None, True)
    q19 = Queue(19, q20, False, None, True)



    #Treatment path: Livmorhals 1
    q27 = Queue(27, None, False, None, True)
    q26 = Queue(26, q27, False, None, True)
    q25 = Queue(25, q26, False, None, True)

    #Treatment path: Livmorhals 2
    q32 = Queue(32, None, False, None, True)
    q31 = Queue(31, q32, False, None, True)
    q30 = Queue(30, q31, False, None, True)
    q29 = Queue(29, q30, False, None, True)
    q28 = Queue(28, q29, False, None, True)

    #Treatment path: Livmorhals 3
    q40 = Queue(40, None, False, None, True)
    q39 = Queue(39, q40, False, None, True)
    q38 = Queue(38, q39, False, None, True)
    q37 = Queue(37, q38, False, None, True)
    q36 = Queue(36, q37, False, None, True)
    q35 = Queue(35, q36, False, None, True)
    q34 = Queue(34, q35, False, None, True)
    q33 = Queue(33, q34, False, None, True)

    #Treatment path: Eggstokk 1
    q47 = Queue(47, None, False, None, True)
    q46 = Queue(46, q47, False, None, True)
    q45 = Queue(45, q46, False, None, True)
    q44 = Queue(44, q45, False, None, True)
    q43 = Queue(43, q44, False, None, True)
    q42 = Queue(42, q43, False, None, True)
    q41 = Queue(41, q42, False, None, True)

    #Treatment path: Eggstokk 2
    q52 = Queue(52, None, False, None, True)
    q51 = Queue(51, q52, False, None, True)
    q50 = Queue(50, q51, False, None, True)
    q49 = Queue(49, q50, False, None, True)
    q48 = Queue(48, q49, False, None, True)

    q3.potential_treatment_queues = [q14,q19]
    q3.probability_of_treatment_queues = [0.5,0.5,0]

    q8.potential_treatment_queues = [q25,q28,q33]
    q8.probability_of_treatment_queues = [7/24,1/24,2/3,0]

    q13.potential_treatment_queues = [q41,q48]
    q13.probability_of_treatment_queues = [0.3,0.7,0]
    """


    #arr = [q0,q1,q2,q3,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24]#,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13]
    arr = [q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14]#,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24,q25,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36,q37,q38,q39,q40,q41,q42,q43,q44,q45,q46,q47,q48,q49,q50,q51,q52]
    q3.potential_treatment_queues = [q4,q9]
    q3.probability_of_treatment_queues = [0.5,0.5,0]





    #scheduled_appointments = np.full((100,100), 1)
    #scheduled_appointments = np.random.randint(2, size = (70, 10000))


    weeks = 2
    G = None
    E = None
    N = M = 25
    shift = 6

    simulation_horizon = 40
    percentage_increase_in_capacity = 0
    no_show_percentage = 0


    mp.Patient_arrivals_jt = mp.Patient_arrivals_jt * (1 + percentage_increase_in_capacity)
    _, b_variable, _ = mm.optimize_model(weeks = weeks, N_input = N, M_input = M, shift = shift, with_rolling_horizon = False, in_iteration = False, weights = None, G = G,E = E)
    #print(b_variable)
    scheduled_appointments = mf.from_dict_to_matrix_2(b_variable, (53, weeks*7))
    #print(scheduled_appointments)
    s = Simulation(arr, scheduled_appointments[:,:7], no_show_percentage)


    for i in range(simulation_horizon):
        s.next_day()
        if i % 7 == 6 and i > 0:

            E = s.create_E_matrix()
            G = s.create_G_matrix()
            #print("E:",E)
            #print("G:",G)

            _, b_variable, _ = mm.optimize_model(weeks = weeks, N_input = N, M_input = M, shift = shift, with_rolling_horizon = True, in_iteration = False, weights = None, G = G, E = E)
            scheduled_appointments = mf.from_dict_to_matrix_2(b_variable,(53, weeks*7))
            #print(b_variable)
            #print(scheduled_appointments[:,:7])
            #Generating the schedules are for the next week
            s.update_appointments(scheduled_appointments[:,:7])


    print("\nSimulation time:", time.time() - start_time)
    create_graph_1(s)
    create_graph_3(s,"uterin")
    create_graph_3(s,"cervical")
    create_graph_3(s,"ovarian")
    #create_graph_2(s)

    print("Generated:",s.total_patients_generated)
    print("Exited:",s.total_patients_exited)
    if s.total_patients_generated > 0:
        print("Exited/generated:",s.total_patients_exited/s.total_patients_generated)



if __name__ == '__main__':
    print("\n")
    print("******* Run simulation *******")
    print("\n")
    main()
    print("\n")
    print("******* END  *******")
    print("\n")
