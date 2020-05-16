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
    def __init__(self, all_queues, appointments):
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
        self.distirbute_appointments()
        self.total_patients_generated = 0
        self.total_patients_exited = 0


    def distirbute_appointments(self):
        for j in range(self.appointments.shape[0]):
            j_appointments = self.appointments[j]
            queue = self.get_queue(j)
            if queue != None:
                queue.set_appointments(j_appointments)


    def update_queue_development(self):
        self.day_array.append(self.day)
        total_num = 0
        sum_departs = 0
        sum_arrivals = 0
        for queue in self.all_queues:
            arr = self.queue_development[queue.id]
            arr.append(queue.get_number_of_patients_in_queue())
            self.queue_development[queue.id] = arr
            sum_arrivals += queue.num_arrivals
            sum_departs += queue.num_departs
            total_num += queue.get_number_of_patients_in_queue()
        self.total_num_in_queue.append(total_num)
        self.num_arrivals_in_system.append(sum_arrivals)
        self.num_departures_in_system.append(sum_departs)

    def update_queue_days_and_capacities(self):
        for queue in self.all_queues:
            queue.day = self.day
            queue.update_appointment_capacity()
            for patient in queue.patient_list:
                patient.new_day()


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
                sum += queue.generate_arrivals_from_poisson()
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
            if queue.get_next_departure_time() < min_t:
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
            next_departure_queue.handle_depart_event(self.time)
            if next_departure_queue.is_last_queue:
                self.total_patients_exited += 1



class Queue:
    def __init__(self, id, next_Queue, is_incoming_queue, arrival_rate):
        self.id = id
        self.day = 0

        self.patient_list = []
        self.num_in_queue_development = []
        self.is_incoming_queue = is_incoming_queue
        self.is_last_queue = True if next_Queue == None else False

        self.next_departure_time = float('inf')

        self.num_arrivals = 0
        self.num_departs = 0
        self.next_Queue = next_Queue
        self.num_recovery_days = 0
        self.appointments = None
        self.appointment_capacity = 0

        self.expected_number_of_arrivals_per_day = arrival_rate

        #Patients that is transfered between queues
        self.incoming_patients = []

        self.allowed_time = 1000

    def __str__(self):
     return "Queue ("+str(self.id)+"): " + str(self.get_number_of_patients_in_queue())

    #SETTERS
    def set_next_departure_time(self, time):
        self.next_departure_time = time

    def set_appointments(self, appointment_array):
        self.appointments = appointment_array

    def set_recovery_days(self, rec_days):
        self.num_recovery_days = rec_days

    #GETTERS


    def get_number_of_patients_in_queue(self):
        return len(self.patient_list)


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
                p.is_removed_from_queue()
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
            print("Hejhallå")
            self.set_next_departure_time(time)


    #A departure means that a patients leaves the queue and is going into service
    def handle_depart_event(self, time):
        if len(self.patient_list) == 0:
            print("Queue",self.id,"is empty")
            sys.exit()

        departure_patient = self.get_patient_with_highest_m()
        self.remove_patient(departure_patient)
        self.decrease_appointment_capacity()
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






def create_graph(s):
    for key in s.queue_development:
        dev = s.queue_development[key]
        plt.plot(s.day_array,dev,linestyle='-', label="Queue " + str(key))

    plt.xlabel('Days')
    plt.ylabel('Number of patients in queue')
    plt.legend(loc='best')
    plt.title('Simulation')
    plt.grid(True)
    plt.savefig('simulation/sim_figures/simulation_2.png')




def main():
    np.random.seed(1)
    start_time=time.time()

    #Livmor
    q3 = Queue(3, None, False, None)
    q2 = Queue(2, q3, False, None)
    q1 = Queue(1, q2, False, None)
    q0 = Queue(0, q1, True, 0.8)

    #Livmorhals
    q8 = Queue(8, None, False, None)
    q7 = Queue(7, q8, False, None)
    q6 = Queue(6, q7, False, None)
    q5 = Queue(5, q6, False, None)
    q4 = Queue(4, q5, True, 0.6)

    #Eggstokk
    q13 = Queue(13, None, False, None)
    q12 = Queue(12, q13, False, None)
    q11 = Queue(11, q12, False, None)
    q10 = Queue(10, q11, False, None)
    q9 = Queue(9, q10, True, 0.8)

    arr = [q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13]






    weeks = 5
    day_horizon = weeks*mp.week_length

    #scheduled_appointments = np.full((10,100),1)
    #scheduled_appointments = np.random.randint(15, size = (20, 100000))

    _, b_variable, _, _, _ = mm.optimize_model(weeks = weeks, N_input = 10, M_input = 10, shift = weeks * mp.week_length - 1, with_rolling_horizon = False, in_iteration = True, weights = None)
    scheduled_appointments = mf.from_dict_to_matrix(b_variable)

    s = Simulation(arr, scheduled_appointments)


    print("\n")
    print("******* Run simulation *******")
    print("\n")
    for i in range(day_horizon):
        s.next_day()
    print("\nSimulation time:", time.time() - start_time)
    create_graph(s)

    print("Generated:",s.total_patients_generated)
    print("Exited:",s.total_patients_exited)
    if s.total_patients_generated > 0:
        print("Exited/generated:",s.total_patients_exited/s.total_patients_generated)

if __name__ == '__main__':
    main()
