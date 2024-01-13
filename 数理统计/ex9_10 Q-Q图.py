import pandas as pd
import pylab as plt
from scipy.stats import norm, probplot

df = pd.read_csv('data9_5.txt', header=None)
d = df.values[0] # 提取甲班乘积
mu = d.mean(); s = d.std()
sd = sorted(d); n = len(d)
x = (plt.arange(n) + 1/2) / n; yi = norm.ppf(x, mu, s)
plt.rc('font', family='SimHei'); plt.rc('font', size=12)
plt.rc('axes', unicode_minus=False)
plt.subplot(121); plt.plot(yi, sd, 'o', label='Q-Q图')
plt.plot(sd, sd, label='参照直线')
plt.legend()
plt.subplot(122); probplot(d, plot=plt) # 建议不用这种方法，画出来效果跟其他软件不一样
plt.show()