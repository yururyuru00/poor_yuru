from torch_geometric import utils
import torch
from torch_geometric.datasets import Planetoid
import numpy as np
from gensim import models
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image


dataset = Planetoid(root='./data/experiment/', name='Cora')
data = dataset[0]

clf = DecisionTreeClassifier(max_depth=5)
clf = clf.fit(data.x, data.y)
f_importance = clf.feature_importances_
print(f_importance)

'''dot_data = StringIO()  # dotファイル情報の格納先
export_graphviz(clf, out_file=dot_data,
                feature_names=["high", "size", "autolock"],  # 編集するのはここ
                class_names=['0', '1', '2', '3', '4', '5', '6'],
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())'''

'''n_class = torch.max(data.y)+1
n_dim = data.x.size()[1]
X = np.zeros((n_class, n_dim))
for id, obj in enumerate(data.x):
    X[data.y[id]] += obj.cpu().detach().numpy().copy()
X = X.T

print(X)
print(np.shape(X))'''

'''corpus = []
for vec in X.T:
    mini_corpus = []
    for word_id, word_freq in enumerate(vec):
        if(word_freq > 0):
            mini_corpus.append((word_id, word_freq))
    corpus.append(mini_corpus)


# tfidf modelの生成
test_model = models.TfidfModel(corpus)

# corpusへのモデル適用
corpus_tfidf = test_model[corpus]

# 表示
print('===結果表示===')
X_tfidf = np.zeros((n_class, n_dim))
for clus_id, clus in enumerate(corpus_tfidf):
    for (word_id, tfidf_score) in clus:
        X_tfidf[clus_id][word_id] = tfidf_score
print(X_tfidf)
print(np.shape(X_tfidf))
'''
