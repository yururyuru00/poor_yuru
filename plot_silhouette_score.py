import argparse
import glob
import os
import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='any of five datasets')
args = parser.parse_args()

l = glob.glob('../SpectralClustering_withRefine/result/{}/pred_*.csv'.format(args.dataset))
l = sorted(l, key=os.path.getmtime)

silhouette_list = np.zeros(len(l))
for tri, pred in enumerate(l):
    pred_clusters = np.loadtxt(pred)

    if(args.dataset=='cora'):
        idfeature_features_labels = np.genfromtxt("../SpectralClustering_withRefine/data/{0}/{0}.content".format(args.dataset),
                                                                        dtype=np.dtype(str))
        X = np.array([[float(i) for i in vec] for vec in idfeature_features_labels[:, 1:-1]])
    elif(args.dataset=='citeseer'):
        with open("../SpectralClustering_withRefine/data/citeseer/citeseer_content.csv", 'r') as r:
            [dim_size, obj_size] = list(map(int, r.readline().rstrip().split(' ')))
        X = np.zeros((obj_size, dim_size), dtype=np.float)
        dim_idobjct_val = np.genfromtxt("../SpectralClustering_withRefine/data/citeseer/citeseer_content.csv",
                                                            skip_header=1, dtype=(int, int, float))
        for [dim_id, obj_id, val] in dim_idobjct_val:
            X[obj_id][dim_id] = float(val)
    else:
        with open("../data/{}-tweets500_tfidf.txt".format(args.dataset), 'r') as r:
            [dim_size, obj_size] = list(map(int, r.readline().rstrip().split(' ')))
        X = np.zeros((obj_size, dim_size), dtype=np.float)
        dim_idobjct_val = np.genfromtxt("../data/{}-tweets500_tfidf.txt".format(args.dataset),
                                                            skip_header=1, dtype=(int, int, float))
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
    silhouette_list[tri] = silhouette_avg
    plt.axvline(silhouette_avg,color="red",linestyle="--")
    plt.yticks(yticks,cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('silhouette coefficient')
    fig.savefig('../SpectralClustering_withRefine/result/{}/silhouette_score{}.png'.format(args.dataset, tri))
np.savetxt('../SpectralClustering_withRefine/result/{}/silhouette_average.csv'.format(args.dataset), 
                    silhouette_list)