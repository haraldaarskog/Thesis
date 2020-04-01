import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
import pydot

# create state space and initial state probabilities
states = ['Referral','CT', 'MRI', 'Biopsy','Gynecological examination','Outpatient clinic','MDT']

# create transition matrix
# equals transition probability matrix of changing states given a state
# matrix is size (M x M) where M is number of states

q_df = pd.DataFrame(columns=states, index=states)
q_df.loc[states[0]] = [0,1,1,0,0,0,0] #Referral
q_df.loc[states[1]] = [0,0,1,0,1,0,0] #CT
q_df.loc[states[2]] = [0,0,0,1,0,0,0] #MRI
q_df.loc[states[3]] = [0,0,0,0,1,0,0] #Biopsy
q_df.loc[states[4]] = [0,0,0,0,0,1,1] #Gynecological examination
q_df.loc[states[5]] = [0,0,0,0,0,0,1] #Outpatient clinic
q_df.loc[states[6]] = [0,0,0,0,0,0,0] #MDT

#print(q_df)

q = q_df.values
print(q)
#print('\n', q, q.shape, '\n')
#print(q_df.sum(axis=1))

# create a function that maps transition probability dataframe
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges


adj_matr=np.array([
[0,1,0],
[0,0,1]])
edges_wts = _get_markov_edges(q_df)
print(edges_wts)
#pprint(edges_wts)


# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states)
#print(f'Nodes:\n{G.nodes()}\n')

# edges represent transition probabilities
for k, v in edges_wts.items():
    if v>0:
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
#print(f'Edges:')
#print(G.edges(data=True))
nx.drawing.nx_pydot.write_dot(G, 'pet_dog_markov.dot')




(graph,) = pydot.graph_from_dot_file('pet_dog_markov.dot')
graph.write_png('somefile.png')
