import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind

f = open('data9_14.txt')
d = f.readlines()
a = np.array(eval(','.join(d[0].split())))
b = np.array(eval(','.join(d[1].split())))
tstat, p = ttest_ind(a, b, alternative='greater')
n1 = len(a); n2 = len(b)
ta = t.ppf(0.95, n1 + n2 - 2 )
print('检验统计量为:', tstat)
print('p值为:', p)
# 上面调用的是库函数，下面编程计算
n1 = len(a); n2 = len(b)
xa = a.mean(); sa2 = a.var(ddof=1)
xb = b.mean(); sb2 = b.var(ddof=1)
ta = t.ppf(0.95, n1 + n2 - 2)
ts = (xa - xb) / (np.sqrt(((n1 - 1) * sa2 + (n2 - 1) * sb2) / (n1 + n2 - 2)) * np.sqrt(1 / n1 + 1 / n2))
print('检测统计量:', ts)


