import numpy as np
from scipy.interpolate import lagrange

x0 = np.arange(1, 7)
y0 = np.array([16, 18, 21, 17, 15, 12])
p = lagrange(x0, y0) # 求拉格朗日插值插值多项式的系数
print(p)
print('从高次幂到低次幂的系数为：', np.round(p, 4)) # 保留小数
yh = np.polyval(p, [1.5, 2.6]) # 计算函数值
print('预测值为：', np.round(yh, 4))