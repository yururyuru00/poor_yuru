import pandas as pd, networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import urllib.request
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def  make_pred_label_map(pred_: list, label_: list) -> dict:
    pred_ids, label_ids = {}, {}
    for vid, (pred_id, label_id) in enumerate(zip(pred_, label_)):
        if(pred_id in pred_ids):
            pred_ids[pred_id].append(vid)
        else:
            pred_ids[pred_id] = []
            pred_ids[pred_id].append(vid)
        if(label_id in label_ids):
            label_ids[label_id].append(vid)
        else:
            label_ids[label_id] = []
            label_ids[label_id].append(vid)

    pred_pairs, label_pairs = [None for _ in range(3)], [None for _ in range(3)]
    for pred_key, label_key in zip(pred_ids.keys(), label_ids.keys()):
        pred_pairs[pred_key] = set([pair for pair in itertools.combinations(pred_ids[pred_key], 2)])
        label_pairs[label_key] = set([pair for pair in itertools.combinations(label_ids[label_key], 2)])

    table = np.array([[len(label_pair&pred_pair) for label_pair in label_pairs] for pred_pair in pred_pairs])

    G = nx.DiGraph()
    with open('./test.csv', 'r') as r:
        n_preds, n_labels = table.shape[0], table.shape[1]
        G.add_node('s', demand = -n_preds)
        G.add_node('t', demand = n_labels)
        for pred_id in range(n_preds):
            G.add_edge('s', 'p_{}'.format(pred_id), weight=0, capacity=1)
        for source, weights in enumerate(table):
            for target, w in enumerate(weights):
                G.add_edge('p_{}'.format(source), 'l_{}'.format(target), weight=-w, capacity=1)
        for label_id in range(n_labels):
            G.add_edge('l_{}'.format(label_id), 't', weight=0, capacity=1)

    clus_label_map = {}
    result = nx.min_cost_flow(G)
    for i, d in result.items():
        for j, f in d.items():
            if f and i[0]=='p' and j[0]=='l': 
                clus_label_map[int(i[2])] = int(j[2])
    return clus_label_map

url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/SchoolScore.txt'
data = urllib.request.urlretrieve(url, 'SchoolScore.txt')
df = pd.read_csv("SchoolScore.txt", sep='\t', na_values=".") # データの読み込み
kmeans_model = KMeans(n_clusters=3).fit(df.iloc[:, 1:])
labels = kmeans_model.labels_
color_codes = {0:'#00FF00', 1:'#FF0000', 2:'#0000FF'}
colors = [color_codes[x] for x in labels]

pca = PCA()
pca.fit(df.iloc[:, 1:])
PCA(copy=True, n_components=None, whiten=False)
feature = pca.transform(df.iloc[:, 1:])

fig = plt.figure(figsize=(6, 6))
ax1, ax2 = fig.add_subplot(1,2,1), fig.add_subplots(1,2,2)
for x, y, name in zip(feature[:, 0], feature[:, 1], df.iloc[:, 0]):
    plt.text(x, y, name, alpha=0.8, size=10)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, color=colors)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.savefig('./test2.png')