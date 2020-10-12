from torch_geometric.datasets import KarateClub
from torch_geometric import utils
import torch
import random
from torch_geometric.datasets import Planetoid
import itertools
import numpy

dataset = Planetoid(root='./data/experiment/', name='Cora')
data = dataset[0]

pair_list = set(itertools.combinations(range(5), 2))
print(pair_list)
pair_list_2 = set([(10, 10), (11, 11)])
print(pair_list_2)
pairs = list(pair_list) + list(pair_list_2)

for (u, v) in pair_list_2:
    print(u, v)
