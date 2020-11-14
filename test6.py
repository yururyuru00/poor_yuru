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

from DeepGraphClustering import utilities


n_class = 7
dataset = 'Cora'
step = 0
edge = 0

def graph_data_obj_to_nx(data):

    G = nx.Graph()

    num_nodes = data.x.size()[0]
    degree = [0 for w in range(num_nodes)]
    for i,j in data.edge_index.T:
            degree[i] += 1
            degree[j] += 1
    for idx in range(num_nodes):
        G.add_node(idx)
    attr = {idx : {'degree' : degree_idx} for idx, degree_idx in enumerate(degree)}
    nx.set_node_attributes(G, attr)

    edge_index = data.edge_index.cpu().numpy()
    n_edges = edge_index.shape[1]
    for j in range(n_edges):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)
    
    return G

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(dataset == 'KarateClub'):
    data = KarateClub(transform=utilities.GraphAugmenter(step,edge))
else:
    data = Planetoid(root='./data/experiment/', name=dataset,
                        transform=utilities.GraphAugmenter(step,edge))
data_augmented = data[0].to(device)

# Graph visualization before augmentation
G = graph_data_obj_to_nx(data_augmented)
pos = nx.spring_layout(G, k=0.9)
node_size = [ d['degree']*10. for (n,d) in G.nodes(data=True)]
nx.draw_networkx_nodes(G, pos, node_size=node_size)
nx.draw_networkx_edges(G, pos, alpha=0.9, edge_color="c")
plt.savefig('./Karate_step{}_edge{}'.format(step, edge))

print(data_augmented)

# Graph visualization after augmentation


'''def plot_G_contextG_pair(G, context_G, center_substract_idx, c_i):

    nodes_context_G = [n_i for n_i in context_G.nodes]
    color_map = []
    for node in G:
        if node == center_substract_idx:
            color_map.append('red')
        elif node in nodes_context_G:
            color_map.append('blue')
        else:  # when node is in a substract graph G
            color_map.append('orange')

    plt.subplot(111)
    nx.draw(G, node_color=color_map,
            with_labels=True)
    plt.savefig('./data/experiment/test/G_context_G_pair_c{}.png'.format(c_i))'''