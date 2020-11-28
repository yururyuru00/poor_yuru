import os.path as osp
import numpy as np
import random
import torch
import argparse
from torch.nn.parameter import Parameter
from torch import nn
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data
import torchvision.transforms as transforms
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from gensim import models


# from utilities import ExtractAttribute, MaskGraph, spectral_clustering

corpuses = []
for vec in features:
    corpus = []
    idxes = np.where(vec==1.0)[0]
    for idx in idxes:
        corpus.append((int(idx), 1))
    corpuses.append(corpus)

model = models.TfidfModel(corpuses)
corpus_tfidf = model[corpuses]

for doc_idx, doc in enumerate(corpus_tfidf):
    for word_idx, val in doc:
        features[doc_idx][word_idx] = val

with open('./data/cora/features.csv', 'w') as w:
    for idx, feature in enumerate(features):
        w.write('{} '.format(idx))
        for val in feature:
            w.write('{} '.format(val))
        w.write('\n')


with open('./data/cora/edge_index.csv', 'w') as w:
    for u, v in edge_index.T:
        w.write('{} {}\n'.format(u, v))

with open('./data/cora/labels.csv', 'w') as w:
    for idx, label in enumerate(labels):
        w.write('{} {}\n'.format(idx, label))
