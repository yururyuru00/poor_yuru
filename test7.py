import sys, os
sys.path.append(os.path.abspath(".."))
import numpy as np
import random
import torch
from torch.nn.parameter import Parameter
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import KarateClub
from torch_geometric import utils
import itertools
import matplotlib.pyplot as plt
import networkx as nx

from ..DeepGraphClustering.utilities import ExtractAttribute, Mask, spectral_clustering


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='./data/experiment/', name='Cora',
                    transform=(ExtractAttribute(7, 8), Mask(0.15, 0.15)))
data = dataset[0].to(device)

print(data)