import numpy as np
from scipy.stats import t
from scipy.stats import ttest_1samp

a = np.loadtxt('data9_13.txt').flatten()
xb = a.mean(); s = a.std(ddof=1)
n = len(a)
ta = t.ppf(0.95, n - 1) # 拒绝域 ts >= ta
ts, p = ttest_1samp(a, 225, alternative='greater')
print('t统计值为:', ts)
