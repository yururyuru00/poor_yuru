from sklearn.metrics import pairwise_distances
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples

dataset = 'football'

for tri in range(14):
    pred_clusters = np.genfromtxt("../network_clustering/experiment/{0}_pred{1}".format(dataset, tri),
                                                dtype=np.dtype(int))

    with open("../data/{0}-tweets500.txt".format(dataset), 'r') as r:
        [dim_size, obj_size] = list(map(int, r.readline().rstrip().split(' ')))
    X = np.zeros((obj_size, dim_size), dtype=np.float)
    dim_idobjct_val = np.genfromtxt("../data/{0}-tweets500.txt".format(dataset),
                                                            skip_header=1, dtype=np.dtype(int))
    for [dim_id, obj_id, val] in dim_idobjct_val:
        X[obj_id][dim_id] = float(val)
    

    cluster_labels = np.unique(pred_clusters)
    n_clusters=cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, pred_clusters, metric='cosine')
    y_ax_lower, y_ax_upper, yticks= 0, 0, []

    fig = plt.figure(figsize=(17, 30))
    for i,c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[pred_clusters==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i)/n_clusters)
        plt.barh(range(y_ax_lower,y_ax_upper),
                         c_silhouette_vals,
                         height=1.0,
                         edgecolor='none',
                         color=color)
        yticks.append((y_ax_lower+y_ax_upper)/2)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,color="red",linestyle="--")
    plt.yticks(yticks,cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('silhouette coefficient')
    fig.savefig('../network_clustering/experiment/silhouette_{}_{}.png'.format(dataset, tri))