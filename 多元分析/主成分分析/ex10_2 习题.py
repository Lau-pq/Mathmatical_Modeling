import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import zscore

a = np.loadtxt('data10_2.txt')
c = zscore(a, ddof=1) # 数据标准化
md1 = PCA().fit(c) # 构造并拟合模型
print('特征值为:', md1.explained_variance_)
r1 = md1.explained_variance_ratio_ # 提取各主成分的贡献率
print('各主成分贡献率:', r1)
xs1 = md1.components_ # 提出各主成分系数，每行是一个主成分
print('主成分系数:\n', np.round(xs1, 4))
print('累积贡献率:', np.cumsum(r1))

num, vec = np.linalg.eig(c.T @ c)
print(vec)

n1 = 3 # 选取的主成分个数
df1 = c @ (xs1[:n1, :].T) # 计算主成分得分
g1 = df1 @ r1[:n1] # 计算综合评价得分
print('主成分评价得分:', np.round(g1, 4))
ind1 = np.argsort(-g1) # 计算从大到小的地址
ind11 = np.zeros(17); ind11[ind1] = np.arange(1, 18)
print('排序结果:', ind11)
print('-----------------------')

