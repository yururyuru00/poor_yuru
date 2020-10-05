from torch_geometric.datasets import KarateClub
from torch_geometric import utils
import torch
import random
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./data/experiment/', name='Cora')
data = dataset[0]
print(type(data.x))
print(data.y)
data.x = torch.nn.functional.one_hot(data.y, torch.max(data.y)+1).double()
