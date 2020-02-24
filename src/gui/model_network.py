import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
G = nx.DiGraph()






#G.add_edges_from([(1, 2), (1, 3)])
#G.add_node(1)
#G.add_edge(1, 2)
#G.add_node("spam")        # adds node "spam"

#G.add_edge(3, 'm')

#print(G.number_of_nodes())
#G.number_of_edges()
#print(list(G.nodes))



A = np.array([[1,0,1], 
              [1,0,1]])

res=[]

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if A[i,j]==1:
            arr=(i,j)
            res.append(arr)
print(res)
G.add_edges_from(res)

nx.draw(G, with_labels=True, font_weight='bold')


plt.show()