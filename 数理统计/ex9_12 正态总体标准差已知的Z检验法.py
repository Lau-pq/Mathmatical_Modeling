import numpy as np
from scipy.stats import ttest_1samp
from scipy.stats import norm

alpha = 0.05; sigma = 4.2
a = np.array([26.01, 26.00, 25.98, 25.86, 26.32, 25.58, 25.32, 25.89, 26.32, 26.18])
t, p = ttest_1samp(a, 26)
xb = a.mean(); s = a.std(ddof=1)
z = t * s / sigma # 转换为z统计量
za = norm.ppf(1 - alpha/2, 0, 1) # 求上 alpha/2 分位数
print('Z统计量值:', z); print('p值:', p); print('分位数:', za)
