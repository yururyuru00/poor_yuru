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



class Image(object):
    def __init__(self):
        self._width = 300
        self._height = 400

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height

if __name__ == '__main__':
    img = Image()
    img.width = 200
    img.height = 100
    print(img.width)
    print(img.height)