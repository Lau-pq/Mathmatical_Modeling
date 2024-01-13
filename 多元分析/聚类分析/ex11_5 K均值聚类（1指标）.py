import numpy as np
from sklearn.cluster import KMeans

a = np.array([[2, 3, 3.5, 7, 9]]).T
md = KMeans(2).fit(a) # 构建2聚类模型并求解(分成两类)
labels = md.labels_ # 提取聚类标签
centers = md.cluster_centers_ # 每一行是一个聚类中心
print(labels, '\n--------------\n', centers)