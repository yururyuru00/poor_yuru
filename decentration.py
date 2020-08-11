import numpy as np
import glob, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='any of the following datasets {cora, citeseer, football, politicsuk, football}')
args = parser.parse_args()

l = glob.glob('../SpectralClustering_withRefine/result/{}/pred*.csv'.format(args.dataset))
l = sorted(l, key=os.path.getmtime)

decentration_list = []
for pred in l:
    pred = np.genfromtxt(pred, dtype=int)
    n_clusters = np.max(pred)+1
    list_n_clusters = [np.count_nonzero(pred==cluster_id) for cluster_id in range(n_clusters)]
    decentration_list.append(np.var(list_n_clusters))

np.savetxt('../SpectralClustering_withRefine/result/{}/decentration.csv' \
                    .format(args.dataset), decentration_list, fmt='%.5f')
