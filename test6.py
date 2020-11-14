import sys, os
sys.path.append(os.path.abspath(".."))
import numpy
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

    edge_index = data.edge_index.cpu().numpy()
    n_edges = edge_index.shape[1]
    for j in range(n_edges):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)
    
    return G

# parameters
dataset = 'Cora'
sample_label_idxes = [0, 2, 7] # label7 means augmented vertices
color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#000000', '#a65628', '#ffff33']
step = 1000
edge_BA = 1
edge_TF = 10

# data loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(dataset == 'KarateClub'):
    data = KarateClub(transform=GraphAugmenter(step, edge_BA, edge_TF))
else:
    data = Planetoid(root='./data/experiment/', name=dataset,
                        transform=GraphAugmenter(step, edge_BA, edge_TF))
data_augmented = data[0].to(device)

# augmented graph visualization
G = graph_data_obj_to_nx(data_augmented)
v_idxes_of_subgraph = []
for v_id, v_label_id in enumerate(data_augmented.y):
    if(v_label_id in sample_label_idxes):
        v_idxes_of_subgraph.append(v_id)
G_sub = nx.subgraph(G, nbunch=v_idxes_of_subgraph)

plt.figure(figsize = (20, 20))
v_color_list = nx.get_node_attributes(G_sub, 'color').values()
pos = nx.spring_layout(G_sub, seed=42)
nx.draw_networkx_nodes(G_sub, pos, node_size=25, node_color=v_color_list)
nx.draw_networkx_edges(G_sub, pos, width=0.70)
plt.savefig('./{}_step{}_BA{}_TF{}'.format(dataset, step, edge_BA, edge_TF), 
            dpi = 150, bbox_inches="tight")
