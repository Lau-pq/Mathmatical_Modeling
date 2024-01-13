import scipy.cluster.hierarchy as sch
import numpy as np
import pylab as plt

plt.rc('text', usetex=True); plt.rc('font', size=12)
a = np.array([[2, 3, 3.5, 7, 9]]).T
c = sch.linkage(a) # 产生数据矩阵
print(c)
s = ['$\\omega_'+str(i+1)+'$' for i in range(5)]
sch.dendrogram(c, labels=s) # 画聚类树状图
plt.show()
