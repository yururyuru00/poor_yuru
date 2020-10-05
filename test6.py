from torch_geometric.datasets import KarateClub
from torch_geometric import utils
import torch
import random

dataset = KarateClub()

A = utils.to_dense_adj(dataset[0].edge_index)[0, :, :]
adj = utils.dense_to_sparse(A)[0]

print(dataset[0].edge_index)
print(adj)
