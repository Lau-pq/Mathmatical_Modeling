import numpy as np
from sklearn.cluster import KMeans
import pylab as plt
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

a = np.loadtxt('data10_1.txt')
b = (a - a.min(axis=0)) / (a.max(axis=0) - a.min(axis=0))
S = []
K = range(2, len(a))
for i in K:
    md = KMeans(i).fit(b)
    labels = md.labels_ # 聚类标签
    S.append(silhouette_score(b, labels)) # 轮廓系数
plt.plot(K, S, '*-')
plt.show()


md = KMeans(2).fit(a) # 构建2聚类模型并求解(分成两类)
labels = md.labels_ # 提取聚类标签
centers = md.cluster_centers_ # 每一行是一个聚类中心
print(labels, '\n--------------\n', centers)

# 原始数据聚类
plt.rc('text', usetex=True); plt.rc('font', size=12)
plt.figure(figsize=(10, 10))
c = sch.linkage(a, metric='seuclidean') # 产生数据矩阵
print(c)
s = ['$\\omega_'+str(i+1)+'$' for i in range(27)]
sch.dendrogram(c, labels=s) # 画聚类树状图
plt.show() 

