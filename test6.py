import sys, os
sys.path.append(os.path.abspath(".."))
import numpy as np
import random
import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import KarateClub
import itertools
import matplotlib.pyplot as plt
import networkx as nx

from utilities_cp import GraphAugmenter


def graph_data_obj_to_nx(data):

    G = nx.Graph()
    labels = data.y.cuda().cpu().detach().numpy().copy()
    print(labels)

    num_nodes = labels.shape[0]
    for idx in range(num_nodes):
        G.add_node(idx)
    attr = {idx : {'color' : color_list[label_id]} for idx, label_id 
                in enumerate(labels)}
    nx.set_node_attributes(G, attr)

    edge_index = data.edge_index.cuda().cpu().numpy().copy()
    n_edges = edge_index.shape[1]
    for j in range(n_edges):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)
    
    return G

def average_path_length_from_hub(edge_index, obj_size, hub_idx):
    N = np.array([set() for i in range(obj_size)])
    for opponent, obj in edge_index:
        N[obj].add(opponent)
    R = np.array( [ [set() for i in range(obj_size)]
                     for j in range(2) ] )
    Ft = np.array( [ [set() for i in range(obj_size)]
                      for j in range(2) ] )

    for v in range(obj_size):
        R[0][v].add(v)
        Ft[0][v].add(v)

    path_length_sum = 0
    for step in range(1, 100):
        for v in range(len(N)):
            R[1][v] = R[0][v].copy()
            for n in N[v]:
                R[1][v] = R[1][v] | R[0][n]
            Ft[1][v] = R[1][v].copy() - R[0][v]
        
        nodes_from_hub_in_steps = len(Ft[1][hub_idx])
        if(nodes_from_hub_in_steps == 0):
            break
        else:
            path_length_sum += step * nodes_from_hub_in_steps

        for v in range(len(N)):
            R[0][v] = R[1][v].copy()
            Ft[0][v] = Ft[1][v].copy()
            R[1][v].clear()
            Ft[1][v].clear()
    
    avg_path_length = path_length_sum / (obj_size-1)
    return avg_path_length


# parameters
dataset = 'Cora'
sample_label_idxes = [0, 2, 7] # label7 means augmented vertices
color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#000000', '#a65628', '#ffff33']
step = 3300
edge_BA = 1
edge_TF = 5

# data loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(dataset == 'KarateClub'):
    data = KarateClub(transform=GraphAugmenter(step, edge_BA, edge_TF))
else:
    data = Planetoid(root='./data/experiment/', name=dataset,
                        transform=GraphAugmenter(step, edge_BA, edge_TF))
data_augmented = data[0].to(device)
print(data_augmented)

# make networkx object from torch tensor object
G = graph_data_obj_to_nx(data_augmented)

# make subgraph of G
v_idxes_of_subgraph = []
for v_id, v_label_id in enumerate(data_augmented.y):
    if(v_label_id in sample_label_idxes):
        v_idxes_of_subgraph.append(v_id)
G_sub = nx.subgraph(G, nbunch=v_idxes_of_subgraph)

# calc average path length from hub node
nodes_rank = nx.pagerank_scipy(G_sub, alpha=0.85)
hub_node_idx = max((v, k) for k, v in nodes_rank.items())[1]
edge_index = data_augmented.edge_index.cuda().cpu().detach().numpy().copy().T
obj_size = data_augmented.x.size()[0]
avg_length_from_hub = average_path_length_from_hub(edge_index, obj_size, hub_node_idx)


fig = plt.figure(figsize = (20, 20))
v_color_list = nx.get_node_attributes(G_sub, 'color').values()
pos = nx.spring_layout(G_sub, seed=42)
nx.draw_networkx_nodes(G_sub, pos, node_size=25, node_color=v_color_list)
nx.draw_networkx_edges(G_sub, pos, width=0.70)
fig.text(0.74, 0.85, 'avg_path_length : ', size=25)
fig.text(0.8, 0.82, '{:.3f}'.format(avg_length_from_hub), size=25)
plt.savefig('./{}_step{}_BA{}_TF{}'.format(dataset, step, edge_BA, edge_TF), 
            dpi = 150, bbox_inches="tight")
