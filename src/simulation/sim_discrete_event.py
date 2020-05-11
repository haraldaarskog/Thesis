import numpy as np
import matplotlib.pyplot as plt
import time


#TODO: LES LEFTINK

#Master class
class Simulation:
    def __init__(self, all_queues):
        self.all_queues = all_queues
        self.clock = 0
        self.t_arrival_dict = {}
        self.t_depart_dict = {}
        self.total_num_in_queue = [0]
        for queue in self.all_queues:
            self.t_arrival_dict[queue.id] = queue.t_arrival
            self.t_depart_dict[queue.id] = queue.t_depart
        self.queue_development = {}
        for queue in self.all_queues:
            self.queue_development[queue.id] = [queue.num_in_queue]
        self.num_arrivals_in_system = 0
        self.num_exits_in_system = 0

        self.total_wait_array = []
        self.array=[]
        self.times=[0]
        self.max_clock = float('inf')

    def update_queue_development(self):
        total_num = 0
        for queue in self.all_queues:
            arr = self.queue_development[queue.id]
            arr.append(queue.num_in_queue)
            self.queue_development[queue.id] = arr
            total_num += queue.num_in_queue
        self.total_num_in_queue.append(total_num)

    def get_queue(self, queue_id):
        for q in self.all_queues:
            if q.id == queue_id:
                return q


    def advance_time(self):
        next_arrival_queue_id = min(self.t_arrival_dict, key=self.t_arrival_dict.get)
        t_next_arrival_event = self.t_arrival_dict[next_arrival_queue_id]
        q_arrival = self.get_queue(next_arrival_queue_id)

        next_departure_queue_id = min(self.t_depart_dict, key=self.t_depart_dict.get)
        t_next_departure_event = self.t_depart_dict[next_departure_queue_id]
        q_departure = self.get_queue(next_departure_queue_id)

        t_event = min(t_next_arrival_event, t_next_departure_event)

        self.clock = t_event


        if t_next_arrival_event <= t_next_departure_event:
            t_arrival, t_depart = self.get_queue(next_arrival_queue_id).handle_arrival_event(self.clock)
            self.t_arrival_dict[next_arrival_queue_id] = t_arrival
            self.t_depart_dict[next_arrival_queue_id] = t_depart

        else:
            t_depart = q_departure.handle_depart_event(self.clock)
            self.t_depart_dict[next_departure_queue_id] = t_depart

            if not q_departure.is_last_queue:
                #print("A patient is serviced at queue", q_departure.id, "and is about to enter queue", q_departure.next_Queue.id)
                self.t_arrival_dict[q_departure.next_Queue.id] = self.clock



        self.update_queue_development()
        self.times.append(self.clock)



class Queue:
    def __init__(self, id, next_Queue, is_incoming_queue):
        self.id = id
        self.num_in_queue_development = []
        self.num_in_queue = 0
        self.clock = 0
        self.is_incoming_queue = is_incoming_queue
        if next_Queue is None:
            self.is_last_queue = True
        else:
            self.is_last_queue = False

        if is_incoming_queue:
            self.t_arrival = self.generate_next_arrival()
        else:
            self.t_arrival = float('inf')

        self.t_depart = float('inf')
        self.num_arrivals = 0
        self.num_departs = 0
        self.total_wait = 0
        self.next_Queue = next_Queue

    def __str__(self):
     return "Number in queue ("+str(self.id)+"): " + str(self.num_in_queue)

    def handle_arrival_event(self, clock):

        self.num_in_queue += 1
        self.num_arrivals += 1
        if self.num_in_queue <= 1:
            self.t_depart = clock + self.generate_service_time()
        if self.is_incoming_queue:
            self.t_arrival = clock + self.generate_next_arrival()
        else:
            self.t_arrival = float('inf')

        #print("\n")
        #print(clock)
        #print("A patient has arrived in queue", str(self.id)+". Total patients in queue:", self.num_in_queue)

        return self.t_arrival, self.t_depart

    def handle_depart_event(self, clock):

        self.num_in_queue -= 1
        self.num_departs += 1
        if self.num_in_queue > 0:
            self.t_depart = clock + self.generate_service_time()
        else:
            self.t_depart = float('inf')
        #print("\n")
        #print(clock)
        #print("A patient has leaved queue", str(self.id) + ". Total patients in queue:", self.num_in_queue)
        return self.t_depart


    def generate_next_arrival(self):
        return np.random.exponential(1/3)

    def generate_service_time(self):
        return np.random.exponential(1/3)



def create_graph(s):
    for key in s.queue_development:
        dev = s.queue_development[key]
        plt.plot(s.times,dev,linestyle='-', label="Queue "+str(key))

    plt.xlabel('Days')
    plt.ylabel('Number of patients in queue')
    plt.legend(loc='best')
    plt.title('Simulation')
    plt.grid(True)
    plt.savefig('sim_figures/simulation.png')

if __name__ == '__main__':
    start_time=time.time()
    np.random.seed(0)
    q2 = Queue(2, None, False)
    q1 = Queue(1, q2, False)
    q0 = Queue(0, q1, True)

    q4 = Queue(4, None, False)
    q3 = Queue(3, q4, True)


    arr = [q3,q4]

    s = Simulation(arr)

    print("\n")
    print("******* Run simulation *******")
    print("\n")
    for i in range(1000):
        s.advance_time()
    print("Simulation time:", time.time() - start_time)

    create_graph(s)
