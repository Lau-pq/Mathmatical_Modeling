import pandas as pd
import pylab as plt

# 显示 最小值 第一分位数 中位数 第三分位数 最大值
df = pd.read_csv('data9_5.txt', header=None).T
plt.rc('font', family='SimHei'); plt.rc('font', size=12)
plt.boxplot(df, labels=['甲班', '乙班'])
plt.show()