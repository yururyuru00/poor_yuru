from torch_geometric import utils
import torch
from torch_geometric.datasets import Planetoid
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as clus


def normalized(mat):
    for i in range(len(mat)):
        sum_ = np.sum(mat[i])
        for j in range(len(mat[i])):
            if(sum_ == 0):
                mat[i][j] = 0.
            else:
                mat[i][j] = mat[i][j] / sum_


def makeLaplacian(mat):
    L = np.zeros((len(mat), len(mat[0])))
    for d in range(len(mat)):
        for i in range(len(mat)):
            L[d][d] += (mat[i][d] + mat[d][i])
        L[d][d] = L[d][d]/2.0
    for i in range(len(L)):
        for j in range(len(L[0])):
            L[i][j] -= (mat[i][j]+mat[j][i])/2.
    return L


def makeNormLaplacian(mat):
    L = makeLaplacian(mat)
    d = np.zeros(len(mat))
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            d[i] += (mat[i][j] + mat[j][i])
        d[i] = d[i] / 2.
    for i in range(len(L)):
        for j in range(len(L[i])):
            if(d[i] == 0 or d[j] == 0):
                L[i][j] = 0.
            else:
                L[i][j] = (1./np.sqrt(d[i])) * L[i][j] * (1./np.sqrt(d[j]))
    return L


def makeKnn(mat, k):
    knnmat = np.zeros((len(mat), len(mat[0])))
    for i in range(len(mat)):
        arg = np.argsort(-mat[i])
        for top in range(k):
            knnmat[i][arg[top]] = mat[i][arg[top]]
    return knnmat


dataset = Planetoid(root='./data/experiment/', name='Cora')
data = dataset[0]

obj_size = data.x.size()[0]
features = data.x.cpu().detach().numpy().copy()
S1 = np.zeros((obj_size, obj_size))
for i in range(obj_size):
    for j in range(i+1, obj_size):
        S1[i][j] = np.dot(features[i], features[j]) / \
            (np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
        S1[j][i] = S1[i][j]
for d in range(len(S1)):
    S1[d][d] = 0.

S1 = makeKnn(S1, 80)
normalized(S1)

S2 = utils.to_dense_adj(data.edge_index)[0]
S2 = S2.cpu().detach().numpy().copy()
normalized(S2)

S = (S1+S2)/2.

Ls = makeNormLaplacian(S)
eigen_val, eigen_vec = eigsh(Ls, 7, which="SM")

k_means = KMeans(n_clusters=7, n_init=10, tol=0.0000001)
k_means.fit(eigen_vec)
labels = data.y.cpu().detach().numpy().copy()
print(clus.adjusted_rand_score(labels, k_means.labels_))
print(clus.adjusted_mutual_info_score(
    labels, k_means.labels_, "arithmetic"))
