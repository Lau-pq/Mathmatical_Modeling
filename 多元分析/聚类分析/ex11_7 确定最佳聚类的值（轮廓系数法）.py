import numpy as np
from sklearn.cluster import KMeans
import pylab as plt
from sklearn.metrics import silhouette_score

a = np.loadtxt('data11_2.txt')
b = (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0))
S = []
K = range(2, len(a))
for i in K:
    md = KMeans(i).fit(b)
    labels = md.labels_ # 聚类标签
    S.append(silhouette_score(b, labels)) # 轮廓系数
plt.plot(K, S, '*-')
plt.show()