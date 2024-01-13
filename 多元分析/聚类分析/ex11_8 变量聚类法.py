import pandas as pd
import scipy.cluster.hierarchy as sch
import pylab as plt
import numpy as np

a = pd.read_excel('data11_8.xlsx', header=None)
b = a.values.T # 空为nan
b = np.triu(b, k=1) # 去对角线上方的元素 空为0
r = b[np.nonzero(b)] # 去掉0
d = 1 - abs(r)
z = sch.linkage(d, 'complete') # 最大距离
sch.dendrogram(z, labels=range(1, 15))
plt.show()