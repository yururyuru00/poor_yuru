import torch
import os
import sys
sys.path.append('D:/python/GCN/DeepGraphClustering/data/utilities')
import numpy as np
import matplotlib.pyplot as plt


'''pseudo = np.loadtxt('../GCN/DeepGraphClustering/data/experiment/epochs200_skips50/pseudo#0_mapped.csv')
label = np.loadtxt('../GCN/DeepGraphClustering/data/experiment/label.csv')

map1, map2 = [0 for _ in range(7)], [0 for _ in range(7)]
for p, l in map(lambda x : (int(x[0]), int(x[1])), zip(pseudo, label)):
    map1[p] += 1
    map2[l] += 1
print(map1)
print(map2)
fig = plt.figure(figsize=(35, 35))
ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)
ax1.bar([i for i in range(len(map1))], map1)
ax2.bar([i for i in range(len(map2))], map2)
plt.savefig('../GCN/DeepGraphClustering/data/experiment/epochs200_skips50/cluster_size')'''

fig = plt.figure(figsize=(35,17))
ax = [fig.add_subplot(1, 3, 1),fig.add_subplot(1, 3, 2),fig.add_subplot(1, 3, 3)]
min__, max__ = 100, -100
for i, epoch in enumerate([1,20,49]):
    pred = np.loadtxt('../GCN/DeepGraphClustering/data/experiment/epochs200_skips1_l2norm/predlabel_mapped_epoch#{}.csv'.format(epoch))
    min_, max_ = np.min(pred), np.max(pred)
    if(min_ < min__):
        min__ = min_
    if(max_ > max__):
        max__ = max_
for i, epoch in enumerate([1,20,49]):
    pred = np.loadtxt('../GCN/DeepGraphClustering/data/experiment/epochs200_skips1_l2norm/predlabel_mapped_epoch#{}.csv'.format(epoch))
    for p in pred[::700]:
        ax[i].plot(p)
    ax[i].set_ylim(min__, max__)
plt.savefig('../GCN/DeepGraphClustering/data/experiment/epochs200_skips1_l2norm/SoftorHard_perepoch_4sample.png')