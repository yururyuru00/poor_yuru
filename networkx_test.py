import networkx as nx
import numpy as np

G = nx.Graph()
x = np.random.randn(5, 3)
print(x, end='\n\n')
for idx, attrs in enumerate(x):
    G.add_node(idx)
    attr = {idx: {dim_idx: attr for dim_idx, attr in enumerate(attrs)}}
    nx.set_node_attributes(G, attr)

print(len(G.nodes[2]))
