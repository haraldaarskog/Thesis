import numpy as np
import matplotlib.pyplot as plt
import time


#TODO: LES LEFTINK

#Master class
class Simulation:
    def __init__(self, all_queues, appointments):
        self.all_queues = all_queues
        self.day = -1
        self.time = 0
        self.t_arrival_dict = {}
        self.t_depart_dict = {}
        self.total_num_in_queue = [0]
        self.queue_development = {}
        for queue in self.all_queues:
            self.queue_development[queue.id] = [queue.num_in_queue]

        self.num_arrivals_in_system = [0]
        self.num_departures_in_system = [0]

        self.total_wait_array = []

        self.day_array=[0]

        self.appointments = appointments
        self.distirbute_appointments()


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
            arr.append(queue.num_in_queue)
            self.queue_development[queue.id] = arr
            sum_arrivals += queue.num_arrivals
            sum_departs += queue.num_departs
            total_num += queue.num_in_queue
        self.total_num_in_queue.append(total_num)
        self.num_arrivals_in_system.append(sum_arrivals)
        self.num_departures_in_system.append(sum_departs)

    def update_queue_days_and_capacities(self):
        for queue in self.all_queues:
            queue.day = self.day
            queue.update_appointment_capacity()


    #setting all t_arrivals to zero
    def update_arrival_times(self):
        for queue in self.all_queues:
                queue.set_next_arrival_time(float('inf'))


    def load_patients_in_service_into_next_queue(self):
        for queue in self.all_queues:
            queue.update_patients_in_queue_from_dict(self.day)

    def update_departure_times(self):
        for queue in self.all_queues:
            if queue.num_in_queue > 0 and queue.get_appointment_capacity() > 0:
                queue.set_next_departure_time(0)
            else:
                queue.set_next_departure_time(float('inf'))


    def get_queue(self, queue_id):
        for q in self.all_queues:
            if q.id == queue_id:
                return q
        return None


    def update_new_arrivals(self):
        for queue in self.all_queues:
            queue.handle_new_arrivals(self.time)



    def next_day(self):
        self.day += 1
        print("\nDAY", self.day)
        self.update_queue_days_and_capacities()
        self.load_patients_in_service_into_next_queue()
        self.update_arrival_times()
        self.update_departure_times()
        self.update_new_arrivals()

        self.time = 0
        while self.remaining_events_this_day():
            self.advance_time()
        self.update_queue_development()



    def remaining_events_this_day(self):
        arr = self.find_next_arrival()
        dep = self.find_next_departure()
        if arr != None or dep != None:
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

        #returning the queue with the lowest time
    def find_next_arrival(self):
        return_queue = None
        min_t = float('inf')
        for queue in self.all_queues:
            if queue.get_arrival_time() < float('inf'):
                if min_t > queue.get_arrival_time():
                    min_t = queue.get_arrival_time()
                    return_queue = queue
        return return_queue


    def advance_time(self):
        print("\nTIME IS ADVANCED (t = " + str(self.time) +")")
        #Next arrival
        next_arrival_queue = self.find_next_arrival()
        if next_arrival_queue == None:
            next_arrival_time = float('inf')
        else:
            next_arrival_time = next_arrival_queue.get_arrival_time()

        #Next departure
        next_departure_queue = self.find_next_departure()


        if next_departure_queue == None:
            next_departure_time = float('inf')
        else:
            next_departure_time = next_departure_queue.get_next_departure_time()

        print("Arrival queue:",next_arrival_queue,"at t =", next_arrival_time)
        print("Departure queue", next_departure_queue,"at t =", next_departure_time)

        if next_arrival_time == float('inf') and next_departure_time == float('inf'):
            print("Ingenting skjer nå")
            return

        t_event = min(next_arrival_time, next_departure_time)
        self.time = t_event


        if next_arrival_time <= next_departure_time:
            next_arrival_queue.handle_arrival_event(self.time)

        else:
            next_departure_queue.handle_depart_event(self.time)



class Queue:
    def __init__(self, id, next_Queue, is_incoming_queue):
        self.id = id
        self.day = 0

        self.num_in_queue_development = []
        self.num_in_queue = 0
        self.is_incoming_queue = is_incoming_queue

        self.is_last_queue = True if next_Queue == None else False


        self.next_departure_time = float('inf')
        self.next_arrival_time = float('inf')

        self.num_arrivals = 0
        self.num_departs = 0
        self.next_Queue = next_Queue
        self.num_recovery_days = 0
        self.appointments = None
        self.appointment_capacity = 0
        self.arrival_dict = {}

        self.resource_dict={}

    def __str__(self):
     return "Queue ("+str(self.id)+"): " + str(self.num_in_queue)

    def get_patients_from_arrival_dict(self, day):
        if day in self.arrival_dict.keys():
            return self.arrival_dict[day]
        else:
            return 0

    def update_patients_in_queue_from_dict(self, day):
        print(self.get_patients_from_arrival_dict(day), "patients arrived in queue", self.id, "from a previous queue")
        self.num_in_queue += self.get_patients_from_arrival_dict(day)

    def set_next_departure_time(self, time):
        self.next_departure_time = time

    def set_next_arrival_time(self, time):
        self.next_arrival_time = time

    def set_appointments(self, appointment_array):
        self.appointments = appointment_array

    def update_appointment_capacity(self):
        self.appointment_capacity = self.appointments[self.day]
        print("Capacity of queue:",self.id,"is:",self.appointment_capacity)

    def decrease_appointment_capacity(self):
        if self.appointment_capacity > 0:
            self.appointment_capacity -= 1
        else:
            print("Kapasiteten er allerede 0")

    def get_appointment_capacity(self):
        return self.appointment_capacity

    def set_recovery_days(self, rec_days):
        self.num_recovery_days = rec_days

    def set_arrival_day(self, arrival_day):
        if arrival_day not in self.arrival_dict.keys():
            self.arrival_dict[arrival_day] = 1
        else:
            self.arrival_dict[arrival_day] = self.arrival_dict[arrival_day] + 1

    def generate_service_time(self):
        return np.random.exponential(1/3)

    def allowed_time(self):
        return 2

    def get_arrival_time(self):
        return self.next_arrival_time

    def get_next_departure_time(self):
        return self.next_departure_time


    #check
    def handle_arrival_event(self, time):
        self.num_in_queue += 1
        self.num_arrivals += 1
        print("EVENT: Arr: q("+str(self.id)+"), in queue: " + str(self.num_in_queue)+", t =", time)
        next_service_time = time + self.generate_service_time()
        if self.num_in_queue <= 1 and self.get_appointment_capacity() > 0 and next_service_time < self.allowed_time():
            self.set_next_departure_time(next_service_time)
        self.next_arrival_time = float('inf')

    #check
    def handle_new_arrivals(self, time):
        if self.is_incoming_queue and (self.day % 7) == 0:
            new_arrivals = self.generate_new_arrivals()
            self.num_arrivals += new_arrivals
            if new_arrivals > 0 and self.num_in_queue == 0 and self.get_appointment_capacity() > 0:
                self.num_in_queue += new_arrivals
                next_service_time = time + self.generate_service_time()
                if next_service_time < self.allowed_time():
                    self.set_next_departure_time(time + self.generate_service_time())
                else:
                    self.set_next_departure_time(float('inf'))

            else:
                self.num_in_queue += new_arrivals
            print("EVENT: Big arrival:", new_arrivals, "in queue:", self.id)


    def generate_new_arrivals(self):
        mean = 10
        std_deviation = 6
        new_arrivals = np.round(np.random.normal(mean, std_deviation, 1))[0]
        if new_arrivals < 0:
            return 0
        else:
            return new_arrivals


    def handle_depart_event(self, time):
        self.num_in_queue -= 1
        self.num_departs += 1
        self.decrease_appointment_capacity()
        print("EVENT: Dep: q("+str(self.id)+"), remaining: "+ str(self.num_in_queue) + ", t =", time)
        if self.num_in_queue > 0 and self.get_appointment_capacity() > 0:
            next_service_time = time + self.generate_service_time()

            #Hvis det er mer tid tilgjengelig, gjennomfør neste time samme dag
            if next_service_time < self.allowed_time():
                self.set_next_departure_time(next_service_time)
            #Hvis ikke mer tid, sett tid til inf
            else:
                self.set_next_departure_time(float('inf'))

        else:
            self.set_next_departure_time(float('inf'))

        if not self.is_last_queue:
            self.next_Queue.set_arrival_day(self.day + self.num_recovery_days)
            if self.num_recovery_days > 0:
                self.next_Queue.set_next_arrival_time(float('inf'))
            else:
                self.next_Queue.set_next_arrival_time(time)



def create_graph(s):
    for key in s.queue_development:
        dev = s.queue_development[key]
        plt.plot(s.day_array,dev,linestyle='-', label="Queue "+str(key))

    plt.xlabel('Days')
    plt.ylabel('Number of patients in queue')
    plt.legend(loc='best')
    plt.title('Simulation')
    plt.grid(True)
    plt.savefig('sim_figures/simulation_2.png')










def main():
    start_time=time.time()
    np.random.seed(0)
    q2 = Queue(2, None, False)
    q1 = Queue(1, q2, False)
    q0 = Queue(0, q1, True)

    q4 = Queue(4, None, False)
    q3 = Queue(3, q4, True)


    arr = [q0,q1,q2,q3,q4]

    """
    scheduled_appointments = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    """

    day_horizon = 1
    scheduled_appointments = np.random.randint(3, size = (5, day_horizon))

    s = Simulation(arr, scheduled_appointments)

    print("\n")
    print("******* Run simulation *******")
    print("\n")
    for i in range(day_horizon):
        s.next_day()


    print("\nSimulation time:", time.time() - start_time)
    create_graph(s)

if __name__ == '__main__':
    #main()

    sum = 0
    count = 0
    for i in range(1000):
        count += 1
        a = np.round(np.random.exponential(1/3) * 7)
        print(a)
