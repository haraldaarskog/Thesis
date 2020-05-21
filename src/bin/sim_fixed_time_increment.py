import numpy as np
import matplotlib.pyplot as plt
import time


#TODO: LES LEFTINK

#Master class
class Simulation:
    def __init__(self, all_queues):
        self.all_queues = all_queues
        self.day = -1
        self.t_arrival_dict = {}
        self.t_depart_dict = {}
        self.total_num_in_queue = [0]

        self.queue_development = {}

        for queue in self.all_queues:
            self.queue_development[queue.id] = [queue.num_in_queue]

        self.num_arrivals_in_system = 0
        self.num_exits_in_system = 0

        self.total_wait_array = []
        self.array=[]
        self.times=[0]
        self.max_day = float('inf')

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

    def update_system(self):
        clock = 0

        for queue in self.all_queues:
            queue.update_queue(self.day)

    def advance_time(self):
        self.day += 1
        self.update_system()
        self.update_queue_development()
        self.times.append(self.day)



class Queue:
    def __init__(self, id, next_Queue, is_incoming_queue):
        self.id = id
        self.num_in_queue = 0
        self.day = 0
        self.is_incoming_queue = is_incoming_queue
        if next_Queue is None:
            self.is_last_queue = True
        else:
            self.is_last_queue = False

        self.t_next_departure = float('inf')
        self.num_arrivals = 0
        self.num_departs = 0
        self.total_wait = 0
        self.next_Queue = next_Queue

    def __str__(self):
     return "Number in queue ("+str(self.id)+"): " + str(self.num_in_queue)

    def update_queue(self, day):
        if self.is_incoming_queue and (day % 7) == 0:
            self.handle_new_arrivals()

        if self.num_in_queue > 0:
            self.handle_departure(0)


    def handle_new_arrivals(self):
        new_arrivals = self.generate_new_arrivals()
        self.num_in_queue += new_arrivals
        self.num_arrivals += new_arrivals
        print(new_arrivals, "new patients arrived in queue",self.id)
        if self.num_in_queue == 0 and new_arrivals > 0:
            self.handle_departure()

    def handle_single_arrival(self, time):
        self.num_in_queue += 1
        self.handle_departure(time)
        print("A patient has arrived in queue", self.id,"at time",time,"and there are now",self.num_in_queue,"in this queue")


    def handle_departure(self, time):
        time = time
        max_time = 8
        while self.num_in_queue > 0:
            print(self.num_in_queue)
            service_time = 10 * self.generate_service_time()
            time += service_time
            self.t_next_departure = time
            if time > max_time:
                break
            elif not self.is_last_queue:
                self.num_in_queue -= 1
                self.next_Queue.handle_single_arrival(time)
            else:
                self.num_in_queue -= 1



    def generate_new_arrivals(self):
        mean = 20
        std_deviation = 2
        return np.round(np.random.normal(mean, std_deviation, 1))[0]

    def generate_service_time(self):
        return np.random.exponential(1/4)



def create_graph(s):
    for key in s.queue_development:
        dev = s.queue_development[key]
        plt.plot(s.times,dev,linestyle='-', label="Queue "+str(key))

    plt.xlabel('Days')
    plt.ylabel('Number of patients in queue')
    plt.legend(loc='best')
    plt.title('Simulation')
    plt.grid(True)
    plt.savefig('sim_figures/simulation_fixed_time_increment.png')

if __name__ == '__main__':
    start_time=time.time()
    np.random.seed(0)
    #q2 = Queue(2, None, False)
    #q1 = Queue(1, q2, False)
    #q0 = Queue(0, q1, True)

    q4 = Queue(4, None, False)
    q3 = Queue(3, q4, True)


    arr = [q3,q4]

    s = Simulation(arr)

    print("\n")
    print("******* Run simulation *******")
    print("\n")
    for i in range(1):
        s.advance_time()
    #print(s.queue_development)
    #print(s.times)
    print("Simulation time:", time.time() - start_time)

    create_graph(s)
