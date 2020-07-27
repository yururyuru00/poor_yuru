import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DummyModel(nn.Module):
    def __init__(self, din, dout):
        super(DummyModel, self).__init__()

        self.aff = nn.Linear(din, dout)

    def forward(self, x):
        x = F.tanh(self.aff(x))
        return x

class DummyModel2(nn.Module):
    def __init__(self, base, nclass):
        super(DummyModel2, self).__init__()

        self.aff = base.aff
        self.classifier = nn.Linear(self.aff.out_features, nclass)

    def forward(self, x):
        x_ = F.tanh(self.aff(x))
        l = self.classifier(x_)
        l = F.log_softmax(l, dim=1)
        return l, x_