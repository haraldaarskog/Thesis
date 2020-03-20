import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
G = nx.MultiDiGraph()


#G.add_edges_from([(1, 2)])

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
#G.add_edges_from(res)
G.add_edges_from([(1, 2),(1,3),(1,3)])
#G.add_edge(1,4,weight=7, capacity=15, length=342.7)

nx.draw(G, arrows=True,with_labels=True, font_weight='bold')
#nx.draw_networkx_edge_labels(G,pos = nx.spring_layout(G),edge_labels={(1,2):500,(1,0):500})
plt.show()
