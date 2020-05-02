"""
import queueing_tool as qt
import networkx as nx

g = qt.generate_random_graph(200, seed=3)
q = qt.QueueNetwork(g, seed=3)
q.max_agents = 20000
q.initialize(100)
q.simulate(10000)

pos = nx.nx_agraph.graphviz_layout(g.to_undirected(), prog='fdp')
scatter_kwargs = {'s': 30}
q.draw(pos=pos, scatter_kwargs=scatter_kwargs, bgcolor=[0,0,0,0],figsize=(10, 16), fname='fig.png',bbox_inches='tight')
"""


import queueing_tool as qt
import numpy as np
import sys

adja_list = {0: [1, 4], 1: [2,3], 4:[5,6]}

#edge_list = {0: {1: 1}, 1: {k: 2 for k in range(2, 22)}}
edge_list = {0: {1: 1, 4:1}, 1: {2: 2, 3:2}, 4: {5:2, 6:2}}

g = qt.adjacency2graph(adjacency=adja_list, edge_type=edge_list)

def rate(t):
    return 25 + 350 * np.sin(np.pi * t / 2)**2

def arr_f(t):
    return qt.poisson_random_measure(t, rate, 375)


def ser_f(t):
    return t + np.random.exponential(0.2 / 2.1)


q_classes = {1: qt.QueueServer, 2: qt.QueueServer}
q_args    = {
    1: {
        'arrival_f': arr_f,
        'service_f': lambda t: t,
        'AgentFactory': qt.GreedyAgent},
    2: {
        'num_servers': 1,
        'service_f': ser_f}
}
qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, seed=13)


qn.g.new_vertex_property('pos')
pos = {}
for v in qn.g.nodes():
    if v == 0:
        pos[v] = [0, 0.8]
    elif v == 1:
        pos[v] = [-1, 0.4]
    elif v == 4:
        pos[v] = [1, 0.4]
    else:
        pos[v] = [-5. + (v - 2.0) / 2, 0]

qn.g.set_pos(pos)


#qn.draw(figsize=(12, 3))
#qn.draw(fname="store.png", figsize=(12, 3), bbox_inches='tight')
qn.initialize(edge_type=1)
qn.animate(t=2, figsize=(4, 4))
#qn.simulate(t=1.9)
#print(qn.num_events)
#qn.draw(fname="sim.png", figsize=(12, 3), bbox_inches='tight')

"""
qn.start_collecting_data()
qn.simulate(t=1.9)
data = qn.get_queue_data(edge=(1, 3))
print(data)
"""
