# 折线图
import pandas as pd
import pylab as plt
plt.rc('font', family='SimHei') # 用于正常显示中文标签
plt.rc('font', size=12) # 设置显示字体大小
a = pd.read_excel("data2_52.xlsx", header=None)
b = a.values # 提取其中的数据
x = b[0]; y = b[1:]
plt.plot(x, y[0], '-*b', label='钻石')
plt.plot(x, y[1], '--dr', label='铂金')
plt.xlabel('月份'); plt.ylabel('每月销量')
plt.legend(loc='upper left')
plt.grid() 
plt.show()

# Pandas 结合 Matplotlib 进行数据可视化
b = a.T; b.plot(kind='bar'); plt.legend(['钻石', '铂金'])
plt.xticks(range(6), b[0], rotation=0)
plt.xlabel("月份"); plt.ylabel('数量')
plt.show()

# 子图
# 一个柱状图 一个饼图 曲线 y=sin(10x)/x
import pylab as plt
import numpy as np
plt.rc('text', usetex=True) # 调用 tex 字库
y1 = np.random.randint(2, 5, 6)
y1 = y1 / sum(y1)
plt.subplot(221)
str = ['Apple', 'grape', 'peach', 'pear', 'banana', 'pineapple']
plt.barh(str, y1) # 水平柱状图
plt.subplot(222)
plt.pie(y1, labels=str) # 饼图
plt.subplot(212)
x2 = np.linspace(0.01, 10, 100)
y2 = np.sin(10 * x2) / x2
plt.plot(x2, y2)
plt.xlabel('$x$')
plt.ylabel('$\\mathrm{sin}(10x)/x$')
plt.show()





