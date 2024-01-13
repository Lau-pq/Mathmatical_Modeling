import numpy as np
import scipy.cluster.hierarchy as sch
import pylab as plt

plt.rc('text', usetex=True); plt.rc('font', size=12)
a = np.array([[2, 3, 3.5, 7, 9]]).T
n = len(a)
c = sch.linkage(a, 'complete', 'mahalanobis') # 最大距离 马氏距离
s = ['$\\omega_'+str(i+1)+'$' for i in range(n)]
sch.dendrogram(c, labels=s)
plt.show()
n0 = eval(input('请输入聚类的系数n0:\n'))
cluster = sch.fcluster(c, t = n0, criterion='maxclust')
print('聚类的结果为:', cluster)

