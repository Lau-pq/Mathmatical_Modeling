import numpy as np
import pylab as plt

x0 = np.arange(1, 7)
y0 = np.array([16, 18, 21, 17, 15, 12])
A = np.vander(x0) # 生成范德蒙德矩阵
p = np.linalg.inv(A) @ y0 # 求插值多项式系数
print('从高次幂到低次幂的系数为：', np.round(p, 4)) # 保留小数
yh = np.polyval(p, [1.5, 2.6]) # 计算函数值
print('预测值为：', np.round(yh, 4))
plt.plot(x0, y0, 'o')
xt = np.linspace(1, 6, 100)
plt.plot(xt, np.polyval(p, xt)) # 画插值曲线
plt.show()
