import numpy as np
import matplotlib.pyplot as plt
import time
import main_model as mm
import model_functions as mf
import model_parameters as mp
import sys
import names



#TODO: LES LEFTINK
#TODO: Implementer en metode som beregner total ventetid i systemet. Og gj.sn ventetid/pas
#TODO: sannsynligheten for å måtte vente. Antall som ved køankomst har ventet/køankomester


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
        self.serviced_patient_history = {}
        self.patient_exit_list = []


    def calculate_resource_usage(self):
        number_of_resources = mp.L_rt.shape[0]
        res_dict = {}
        for queue in self.all_queues:
            res_use_queue = queue.resource_usage
            for key in res_use_queue:
                if key not in res_dict:
                    res_dict[key] = np.zeros(number_of_resources)
                res_dict[key] += res_use_queue[key]
        return res_dict

    def calculate_resource_usage_day(self, day):
        number_of_resources = mp.L_rt.shape[0]
        res_array = np.zeros(mp.L_rt.shape[0])
        for queue in self.all_queues:
            res_use_queue = queue.resource_usage
            if day not in res_use_queue:
                continue
            else:
                queue_resources = res_use_queue[day]
                res_array += queue_resources
        return res_array

    def check_if_available_resources(self, queue):
        resources = queue.resource_dict.keys()
        resource_usage = self.calculate_resource_usage_day(self.day)
        for r in resources:
            res_cap = mp.L_rt[r, 0]
            resource_usage_now = resource_usage[r]
            if resource_usage_now + queue.activity_duration > res_cap:
                return False
        return True



    def calculate_waiting_times(self):
        number_of_exits = len(self.patient_exit_list)
        if number_of_exits == 0:
            return 0
        sum = 0
        for patient in self.patient_exit_list:
            sum += patient.number_of_days_in_system
        return sum/number_of_exits




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
                    g_dict[j,t,m] = g_dict[j,t,m] + 1
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
                            if (queue_id, n + 1, m + 1) not in e_dict.keys():
                                e_dict[queue_id, n + 1, m + 1] = 1
                            else:
                                e_dict[queue_id, n + 1, m + 1] = e_dict[queue_id, n + 1, m + 1] + 1
        return e_dict


    def distribute_appointments(self):
        for j in range(self.appointments.shape[0]):
            for q in self.all_queues:
                q.prob_of_no_show = self.no_show_prob
                if j == q.id and q is not None:
                    j_appointments = self.appointments[j]
                    q.appointments = j_appointments

    def update_appointments(self, scheduled_appointments):
        for j in range(scheduled_appointments.shape[0]):
            for q in self.all_queues:
                if j == q.id and q is not None:
                    j_appointments = scheduled_appointments[j]
                    q.set_appointments(j_appointments)


    def update_queue_development(self):
        self.day_array.append(self.day)
        total_num = 0
        sum_departs = 0
        sum_arrivals = 0
        for queue in self.all_queues:
            arr = self.queue_development[queue.id]
            arr.append(queue.get_number_of_patients_in_patientlist_noshow_incoming())#queue.get_number_of_patients_in_patientlist_noshowlist())
            self.queue_development[queue.id] = arr
            sum_arrivals += queue.num_arrivals
            sum_departs += queue.num_departs
            total_num += queue.get_number_of_patients_in_patientlist_noshow_incoming()#queue.get_number_of_patients_in_patientlist_noshowlist()
        self.total_num_in_queue.append(total_num)
        self.num_arrivals_in_system.append(sum_arrivals)
        self.num_departures_in_system.append(sum_departs)

    def update_queue_days_and_capacities(self):
        for queue in self.all_queues:
            queue.day = self.day
            queue.update_appointment_capacity()
            for p in queue.no_show_list:
                queue.no_show_list.remove(p)
                p.set_arrival_day_in_current_queue(self.day)
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
            #DENNE KAN KOMMENTERES UT HVIS RESSURSPROBLEMER
            if self.check_if_available_resources(queue):
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
        #print("Arrival queue:",next_arrival_queue,"at t =", next_arrival_time)
        #print("Departure queue", next_departure_queue,"at t =", next_departure_time)

        if next_arrival_time <= next_departure_time:
            next_arrival_queue.handle_arrival_event(arrival_patient, self.time)
            next_arrival_queue.remove_patient_from_incoming_list(arrival_patient)

        else:
            dep_patient = next_departure_queue.handle_depart_event(self.time)
            if dep_patient != -1 and dep_patient != -2:
                if self.day in self.serviced_patient_history.keys():
                    self.serviced_patient_history[self.day] = self.serviced_patient_history[self.day] + 1
                else:
                     self.serviced_patient_history[self.day] = 1
            if next_departure_queue.is_last_queue and dep_patient != -1 and dep_patient != -2 and next_departure_queue.is_treatment_queue:
                self.total_patients_exited += 1
                dep_patient.exit_day = self.day
                self.patient_exit_list.append(dep_patient)
                print("A PATIENT LEFT THE SYSTEM")
            if dep_patient == -2:
                print("___________Not enough resources______________")

    def summarize_discharged_from_diagnosis(self):
        sum = 0
        for queue in self.all_queues:
            sum += queue.discharged_from_diagnosis
        return sum


    def get_info_about_every_patient_in_system(self):
        for queue in self.all_queues:
            print("QUEUE:", queue.id)
            if len(queue.patient_list)>0:
                print("Waiting list:")
                for p in queue.patient_list:
                    print(p)
            if len(queue.no_show_list)>0:
                print("No-show list:")
                for p in queue.no_show_list:
                    print(p)
            if len(queue.incoming_patients)>0:
                print("Incoming patient list:")
                for p in queue.incoming_patients:
                    print(p)




class Queue:
    def __init__(self, id, next_Queue, is_incoming_queue, arrival_rate, is_treatment_queue, recovery_days, diagnosis, activity):
        self.id = id
        self.day = 0
        self.is_treatment_queue = is_treatment_queue
        self.diagnosis = diagnosis
        self.activity = activity

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
        self.num_recovery_days = recovery_days
        self.appointments = []
        self.appointment_capacity = 0
        self.appointment_dict = {}

        self.expected_number_of_arrivals_per_day = arrival_rate

        #Patients that is transfered between queues
        self.incoming_patients = []

        self.allowed_time = 24

        self.potential_treatment_queues = []
        self.probability_of_treatment_queues = []

        self.discharged_from_diagnosis = 0

        self.patients_out_of_system = []

        #Resources
        self.resource_usage = {}
        self.resource_dict = mp.activity_resource_dict[self.activity]
        self.activity_duration = mp.activity_duration[self.activity]


    def __str__(self):
     return "Queue ("+str(self.id)+"): " + str(self.get_number_of_patients_in_queue())

    #Resource methods
    def activate_resources(self, service_time):
        if self.day not in self.resource_usage.keys():
            number_of_resources = mp.L_rt.shape[0]
            self.resource_usage[self.day] = np.zeros(number_of_resources)
        for key in self.resource_dict:
            hours_demanded = self.resource_dict[key]
            self.resource_usage[self.day][key] += service_time * hours_demanded



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

    def get_number_of_patients_in_patientlist_noshow_incoming(self):
        return len(self.patient_list) + len(self.no_show_list) + len(self.incoming_patients)


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
        mean_service_time = self.activity_duration
        prob_arr = [0.6*mean_service_time,0.8*mean_service_time,1*mean_service_time,1.2*mean_service_time,1.4*mean_service_time]
        index = np.random.choice(np.arange(0, len(prob_arr)), p=[0.1, 0.2, 0.4, 0.2, 0.1])
        return prob_arr[index]

    def generate_arrivals_from_poisson(self):
        if self.is_weekend():
                new_arrivals = 0
        else:
            new_arrivals = np.random.poisson(self.expected_number_of_arrivals_per_day, 1)[0]
        print(new_arrivals,"new arrivals in queue",self.id)
        for i in range(new_arrivals):
            self.handle_arrival_event(Patient(self.day, names.get_full_name(), self.diagnosis), 0)
        return new_arrivals

    #OTHER

    def is_weekend(self):
        if self.day % 7 == 5 or self.day % 7 == 6:
            return True
        return False

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
        #departure_patient = self.get_FIFO_patient()
        self.decrease_appointment_capacity()
        if self.no_show_is_happening():
            print("We have a no show!")
            self.remove_patient(departure_patient)
            self.no_show_list.append(departure_patient)
            return -1
        self.remove_patient(departure_patient)
        departure_patient.is_removed_from_queue(self.id)
        self.num_departs += 1

        print("EVENT: Dep: q("+str(self.id)+"), remaining: "+ str(self.get_number_of_patients_in_queue()) + ", t =", time)
        actual_service_time = self.generate_service_time()
        service_time_for_departure_patient = time + actual_service_time
        self.activate_resources(actual_service_time)

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

        return departure_patient


    def choose_treatment_path(self):
        index = np.random.choice(np.arange(0, len(self.potential_treatment_queues)+1), p=self.probability_of_treatment_queues)
        #Out of the system
        if index == len(self.potential_treatment_queues):
            print("A PATIENT IS DISCHARGED FROM DIAGNOSIS")
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
        else:
            self.discharged_from_diagnosis += 1
            patient.exit_day = self.day
            self.patients_out_of_system.append(patient)


    #Only patients that are moving forward to another queue
    def transfer_patient_to_next_queue(self, patient, service_time):

        patient.next_queue_arrival_day = self.day + self.num_recovery_days
        if self.num_recovery_days > 0:
            patient.next_queue_arrival_time = 0
        else:
            patient.next_queue_arrival_time = service_time

        self.next_Queue.incoming_patients.append(patient)









class Patient:


    def __init__(self, day, name, diagnosis):
        self.day = day
        self.entering_day = day
        self.exit_day = None
        self.name = name
        self.diagnosis = diagnosis

        self.number_of_days_in_queue = 0
        self.number_of_days_in_system = 0

        self.arrival_time_in_current_queue = float('inf')
        self.arrival_day_in_current_queue = float('inf')
        self.departure_time_from_current_queue = float('inf')

        self.next_queue_arrival_day = float('inf')
        self.next_queue_arrival_time = float('inf')
        self.queue_history = {}

        self.id = id(self)

        self.is_in_queue = None

    def __str__(self):
        return "Name: " + self.name + ", N: " + str(self.number_of_days_in_queue) + ", M: " + str(self.number_of_days_in_system)


    def get_number_of_days_in_system(self):
        return self.number_of_days_in_system

    def get_number_of_days_in_current_queue(self):
        return self.number_of_days_in_queue

    def update_queue_history(self, queue_id):
        self.queue_history[queue_id] = [self.day, self.number_of_days_in_queue,self.number_of_days_in_system]

    def is_removed_from_queue(self, queue_id):
        self.update_queue_history(queue_id)
        self.number_of_days_in_queue = 0
        self.is_in_queue = False

    def is_added_to_a_queue(self, time, day, queue_id):
        self.set_arrival_day_in_current_queue(day)
        self.set_arrival_time_in_current_queue(time)
        #self.update_queue_history(queue_id)
        self.is_in_queue = True

    def new_day(self):
        self.day += 1
        if self.is_in_queue:
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


def create_total_queue_development(s):

    var = 14

    cumsum, moving_aves = [0], [0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i, x in enumerate(s.total_num_in_queue, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= var:
            moving_ave = (cumsum[i] - cumsum[i - var])/var
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    plt.plot(s.day_array, s.total_num_in_queue, linestyle='-',label="Number of patients")
    plt.plot(s.day_array, moving_aves, linestyle='--', label="Moving average")
    plt.xlabel('Days')
    plt.ylabel('Number of patients')
    plt.legend(loc='best')
    plt.title('Total number of patients in queue')
    plt.grid(True)
    plt.savefig('simulation/sim_figures/simulation_total_in_system.png')
    plt.close()

def create_stacked_plot(s):
    #uterine_cancer = [0,1,2,3,13,14,15,16,17,18,19]
    #cerivcal_cancer = [4,5,6,7,8,20,21,22,23,24,25,26,27,28,29,30,31]
    #ovarian_cancer = [9,10,11,12,32,33,34,35,36,37,38,39]

    uterine_flag = True
    cervical_flag = True
    ovarian_flag = True

    for queue in s.all_queues:
        if queue.diagnosis == "uterine":
            if uterine_flag:
                uterine_dev = np.zeros(len(s.queue_development[queue.id]))
                uterine_flag = False
            uterine_dev += s.queue_development[queue.id]


        if queue.diagnosis == "cervical":
            if cervical_flag:
                cervical_dev = np.zeros(len(s.queue_development[queue.id]))
                cervical_flag = False
            cervical_dev += s.queue_development[queue.id]


        if queue.diagnosis == "ovarian":
            if ovarian_flag:
                ovarian_dev = np.zeros(len(s.queue_development[queue.id]))
                ovarian_flag = False
            ovarian_dev += s.queue_development[queue.id]


    labels = ["Uterine", "Cervical", "Ovarian"]
    plt.stackplot(s.day_array, uterine_dev, cervical_dev, ovarian_dev, labels = labels)
    plt.xlabel('Days')
    plt.ylabel('Number of patients')
    plt.legend(loc='upper left')
    plt.title('Stacked plot')
    plt.grid(True)
    plt.savefig("simulation/sim_figures/stacked_plot.png")
    plt.close()

def create_queue_development_all_pathways(s):
    #uterine_cancer = [0,1,2,3,13,14,15,16,17,18,19]
    #cerivcal_cancer = [4,5,6,7,8,20,21,22,23,24,25,26,27,28,29,30,31]
    #ovarian_cancer = [9,10,11,12,32,33,34,35,36,37,38,39]

    uterine_flag = True
    cervical_flag = True
    ovarian_flag = True

    for queue in s.all_queues:
        if queue.diagnosis == "uterine":
            if uterine_flag:
                uterine_dev = np.zeros(len(s.queue_development[queue.id]))
                uterine_flag = False
            uterine_dev += s.queue_development[queue.id]


        if queue.diagnosis == "cervical":
            if cervical_flag:
                cervical_dev = np.zeros(len(s.queue_development[queue.id]))
                cervical_flag = False
            cervical_dev += s.queue_development[queue.id]


        if queue.diagnosis == "ovarian":
            if ovarian_flag:
                ovarian_dev = np.zeros(len(s.queue_development[queue.id]))
                ovarian_flag = False
            ovarian_dev += s.queue_development[queue.id]

    if not uterine_flag:
        plt.plot(s.day_array, uterine_dev, linestyle='-', label="Uterine cancer")
    if not cervical_flag:
        plt.plot(s.day_array, cervical_dev, linestyle='-', label="Cerivcal cancer")
    if not ovarian_flag:
        plt.plot(s.day_array, ovarian_dev, linestyle='-', label="Ovarian cancer")
    plt.xlabel('Days')
    plt.ylabel('Number of patients')
    plt.legend(loc='best')
    plt.title('Number of patients in the different care pathways')
    plt.grid(True)
    plt.savefig("simulation/sim_figures/all_pathways.png")
    plt.close()



def create_cumulative_waiting_times(simulation_model, m_range):
    exit_patients = simulation_model.patient_exit_list
    m_array = np.zeros(m_range)
    m_array_2 = np.arange(0,m_range)
    for patient in exit_patients:
        if patient.entering_day > 50:
            m_value = patient.number_of_days_in_system
            m_array[m_value] += 1

    sum_array = int(np.sum(m_array))
    cum = np.divide(np.cumsum(m_array), sum_array)

    plt.plot(m_array_2, cum, linestyle='-')
    plt.xlabel('Days')
    plt.ylabel('Cumulative amount')
    plt.legend(loc='best')
    plt.title('Cumulative distribution of waiting times')
    plt.grid(True)
    plt.savefig("simulation/sim_figures/simulation_cumulative.png")
    plt.close()

def create_cum_distr_all_diagnosis(simulation_model, m_range):
    exit_patients = simulation_model.patient_exit_list

    uterine_array = np.zeros(m_range)
    cervical_array = np.zeros(m_range)
    ovarian_array = np.zeros(m_range)

    uterine_flag = True
    cervical_flag = True
    ovarian_flag = True

    m_array_2 = np.arange(0,m_range)
    for patient in exit_patients:
        if patient.entering_day > 50:
            if patient.diagnosis == "uterine":
                uterine_array[patient.number_of_days_in_system] += 1
            if patient.diagnosis == "cervical":
                cervical_array[patient.number_of_days_in_system] += 1
            if patient.diagnosis == "ovarian":
                ovarian_array[patient.number_of_days_in_system] += 1

    cum_uterine = np.divide(np.cumsum(uterine_array), np.sum(uterine_array))
    cum_cervical = np.divide(np.cumsum(cervical_array), np.sum(cervical_array))
    cum_ovarian = np.divide(np.cumsum(ovarian_array), np.sum(ovarian_array))

    plt.plot(m_array_2, cum_uterine, linestyle='-', label="Uterine cancer")
    plt.plot(m_array_2, cum_cervical, linestyle='-', label="Cervical cancer")
    plt.plot(m_array_2, cum_ovarian, linestyle='-', label="Ovarian cancer")
    plt.xlabel('Days')
    plt.ylabel('Cumulative amount')
    plt.legend(loc='best')
    plt.title('Cumulative distribution of waiting times')
    plt.grid(True)
    plt.savefig("simulation/sim_figures/simulation_cumulative_all_diagnosis.png")
    plt.close()

def create_capacity_graph(sim, day_horizon):
    used_capacity = np.zeros(day_horizon)
    flag = True
    for queue in sim.all_queues:
        if flag:
            total_capacity = np.zeros(len(queue.appointments))
            flag = False
        total_capacity += queue.appointments
    for key in sim.serviced_patient_history:
        used_capacity[key] = sim.serviced_patient_history[key]

    plt.plot(sim.day_array[1:], total_capacity, linestyle='-', label="Total queue capacity")
    plt.plot(sim.day_array[1:], used_capacity, linestyle='-', label = "Patients serviced")
    plt.xlabel('Days')
    plt.ylabel('Number of patients')
    plt.legend(loc='best')
    plt.title('Simulation')
    plt.grid(True)
    plt.savefig("simulation/sim_figures/simulation_capacity.png")
    plt.close()


def create_total_time_in_system(sim, m_range):
    exit_patients = sim.patient_exit_list
    sum = 0
    m_range = m_range + 30
    m_array = np.zeros(m_range)
    m_array_2 = np.arange(0,m_range)
    for patient in exit_patients:
        if patient.entering_day > 50:
            days_in_system = (patient.exit_day-patient.entering_day)
            m_array[days_in_system] += 1

    sum_array = int(np.sum(m_array))
    cum = np.divide(np.cumsum(m_array), sum_array)

    plt.plot(m_array_2, cum, linestyle='-')
    plt.xlabel('Days')
    plt.ylabel('Cumulative amount')
    plt.legend(loc='best')
    plt.title('Cumulative amount, total time in system')
    plt.grid(True)
    plt.savefig("simulation/sim_figures/simulation_cumulative_overall_time.png")
    plt.close()


def resource_usage_plot(s, active_days, sim_horizon):
    res_dict = s.calculate_resource_usage()
    arr = np.zeros((mp.L_rt.shape[0], sim_horizon))
    days = np.arange(0,active_days)
    for day in res_dict:
        res_usage = res_dict[day]
        for i in range(len(res_usage)):
            arr[i, day] = res_usage[i]

    for r in range(arr.shape[0]):
        plt.plot(days, arr[r,:active_days], linestyle='-', label = str(r))
    plt.xlabel('Days')
    plt.ylabel('Hours')
    plt.legend(loc='best')
    plt.title('Resource usage')
    plt.grid(True)
    plt.savefig("simulation/sim_figures/resource_usage.png")
    plt.close()



def main():
    np.random.seed(0)
    start_time=time.time()

    art = mp.activity_recovery_time

    #Patient demand
    uterine_demand = mp.uterine_demand
    cervical_demand = mp.cervical_demand
    ovarian_demand = mp.ovarian_demand

    #Livmor
    q3 = Queue(3, None, False, None, False, art[3], "uterine", 3)
    q2 = Queue(2, q3, False, None, False, art[2], "uterine", 2)
    q1 = Queue(1, q2, False, None, False, art[1], "uterine", 1)
    q0 = Queue(0, q1, True, uterine_demand/5, False, art[0], "uterine", 0)



    #Livmor_treat 1
    q6 = Queue(6, None, False, None, True, art[9], "uterine", 9)
    q5 = Queue(5, q6, False, None, True, art[1], "uterine", 1)
    q4 = Queue(4, q5, False, None, True, art[6], "uterine", 6)

    #Livmor_treat 2
    q10 = Queue(10, None, False, None, True, art[8], "uterine", 8)
    q9 = Queue(9, q10, False, None, True, art[9], "uterine", 9)
    q8 = Queue(8, q9, False, None, True, art[1], "uterine", 1)
    q7 = Queue(7, q8, False, None, True, art[6], "uterine", 6)

    """

    #Livmorhals
    q8 = Queue(8, None, False, None, False, art[7], "cervical", 7)
    q7 = Queue(7, q8, False, None, False, art[3], "cervical", 3)
    q6 = Queue(6, q7, False, None, False, art[4], "cervical", 4)
    q5 = Queue(5, q6, False, None, False, art[6], "cervical", 6)
    q4 = Queue(4, q5, True, cervical_demand/5, False, art[0], "cervical", 0)

    #Eggstokk
    q12 = Queue(12, None, False, None, False, art[5], "ovarian", 5)
    q11 = Queue(11, q12, False, None, False, art[6], "ovarian", 6)
    q10 = Queue(10, q11, False, None, False, art[2], "ovarian", 2)
    q9 = Queue(9, q10, True, ovarian_demand/5, False, art[0], "ovarian", 0)


    #Treatment path: Livmor 1
    q15 = Queue(15, None, False, None, True, art[9], "uterine", 9)
    q14 = Queue(14, q15, False, None, True, art[1], "uterine", 1)
    q13 = Queue(13, q14, False, None, True, art[6], "uterine", 6)

    #Treatment path: Livmor 2
    q19 = Queue(19, None, False, None, True, art[8], "uterine", 8)
    q18 = Queue(18, q19, False, None, True, art[9], "uterine", 9)
    q17 = Queue(17, q18, False, None, True, art[1], "uterine", 1)
    q16 = Queue(16, q17, False, None, True, art[6], "uterine", 6)



    #Treatment path: Livmorhals 1
    q20 = Queue(20, None, False, None, True,art[9], "cervical", 9)

    #Treatment path: Livmorhals 2
    q23 = Queue(23, None, False, None, True, art[10], "cervical", 10)
    q22 = Queue(22, q23, False, None, True, art[5], "cervical", 5)
    q21 = Queue(21, q22, False, None, True, art[9], "cervical", 9)

    #Treatment path: Livmorhals 3
    q31 = Queue(31, None, False, None, True, art[4], "cervical", 4)
    q30 = Queue(30, q31, False, None, True, art[8], "cervical", 8)
    q29 = Queue(29, q30, False, None, True, art[13], "cervical", 13)
    q28 = Queue(28, q29, False, None, True, art[10], "cervical", 10)
    q27 = Queue(27, q28, False, None, True, art[14], "cervical", 14)
    q26 = Queue(26, q27, False, None, True, art[10], "cervical", 10)
    q25 = Queue(25, q26, False, None, True, art[8], "cervical", 8)
    q24 = Queue(24, q25, False, None, True, art[3], "cervical", 3)

    #Treatment path: Eggstokk 1
    q36 = Queue(36, None, False, None, True, art[8], "ovarian", 8)
    q35 = Queue(35, q36, False, None, True, art[6], "ovarian", 6)
    q34 = Queue(34, q35, False, None, True, art[9], "ovarian", 9)
    q33 = Queue(33, q34, False, None, True, art[8], "ovarian", 8)
    q32 = Queue(32, q33, False, None, True, art[6], "ovarian", 6)

    #Treatment path: Eggstokk 2
    q39 = Queue(39, None, False, None, True, art[8], "ovarian", 8)
    q38 = Queue(38, q39, False, None, True, art[6], "ovarian", 6)
    q37 = Queue(37, q38, False, None, True, art[9], "ovarian", 9)

    q3.potential_treatment_queues = [q13, q16]
    q3.probability_of_treatment_queues = [0.35, 0.15, 0.5]

    q8.potential_treatment_queues = [q20, q21, q24]
    q8.probability_of_treatment_queues = [0.27, 0.03, 0.60, 0.10]

    q12.potential_treatment_queues = [q32, q37]
    q12.probability_of_treatment_queues = [0.25, 0.65, 0.10]

    """
    q3.potential_treatment_queues = [q4, q7]
    q3.probability_of_treatment_queues = [0.35,0.15,0.5]


    arr = [q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]#,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24,q25,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36,q37,q38,q39]

    #Optimization param
    weeks = 2
    G = None
    E = None
    M = 60
    N = int(np.round(M*3/5))
    shift = 6

    #Simulation param
    simulation_horizon = 365
    percentage_increase_in_capacity = 0
    no_show_percentage = 0.05



    number_of_queues = mf.get_total_number_of_queues()
    mp.Patient_arrivals_jt = mp.Patient_arrivals_jt * (1 + percentage_increase_in_capacity)
    _, b_variable, _ = mm.optimize_model(weeks = weeks, N_input = N, M_input = M, shift = shift, with_rolling_horizon = False, in_iteration = False, weights = None, G = G,E = E)

    scheduled_appointments = mf.from_dict_to_matrix_2(b_variable, (number_of_queues, weeks*7))
    scheduled_appointments = scheduled_appointments[:,:7]
    s = Simulation(arr, scheduled_appointments, no_show_percentage)


    for i in range(simulation_horizon):
        s.next_day()

        if i % 7 == 6 and i > 0:

            E = s.create_E_matrix()
            G = s.create_G_matrix()

            create_capacity_graph(s, i + 1)
            _, b_variable, _ = mm.optimize_model(weeks = weeks, N_input = N, M_input = M, shift = shift, with_rolling_horizon = True, in_iteration = False, weights = None, G = G, E = E)
            scheduled_appointments = mf.from_dict_to_matrix_2(b_variable,(number_of_queues, weeks*7))
            scheduled_appointments = scheduled_appointments[:,:7]
            s.update_appointments(scheduled_appointments)


            resource_usage_plot(s,i + 1,simulation_horizon)
            create_cumulative_waiting_times(s, M)
            create_cum_distr_all_diagnosis(s, M)
            create_total_time_in_system(s, M)
            create_queue_development_all_pathways(s)
            #create_stacked_plot(s)
            if i > 6:
                create_total_queue_development(s)








    print("\nSimulation time:", time.time() - start_time)


    print("Generated:",s.total_patients_generated)
    print("Exited:",s.total_patients_exited)
    print("Discharged from diagnosis:",s.summarize_discharged_from_diagnosis())
    if s.total_patients_generated > 0:
        print("Exited/generated:",(s.total_patients_exited + s.summarize_discharged_from_diagnosis())/s.total_patients_generated)
    #print(s.get_info_about_every_patient_in_system())
    print(s.patient_exit_list)
    for p in s.patient_exit_list:
        print(p, p.queue_history)
    print("Avg. waiting time:", s.calculate_waiting_times())



if __name__ == '__main__':
    print("\n")
    print("******* Run simulation *******")
    print("\n")
    main()
    print("\n")
    print("******* END  *******")
    print("\n")
