import pandas as pd
import pylab as plt

df = pd.read_csv('data9_5.txt', header=None)
d = df.T[0] # 提取甲班成绩
plt.rc('font', family='SimHei'); plt.rc('font', size=12)
h = plt.hist(d, density=True, histtype='step', cumulative=True)
print(h)
plt.grid()
plt.show()