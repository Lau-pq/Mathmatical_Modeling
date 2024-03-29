from scipy.interpolate import UnivariateSpline
import pylab as plt
import numpy as np

a = np.loadtxt('E:\竞赛\数学建模\python数学建模\插值与拟合\data7_6.txt')
x0 = a[::3].flatten()  # 提出点的横坐标
y1 = a[1::3].flatten() # 提出下边界的纵坐标
y2 = a[2::3].flatten() # 提出上边界的纵坐标
plt.plot(x0, y1, '*-'); plt.plot(x0, y2, '.-')
f1 = UnivariateSpline(x0, y1) # 计算三次样条函数
f2 = UnivariateSpline(x0, y2)
d1 = f1.derivative(1)
d2 = f2.derivative(1)
x = np.linspace(x0[0], x0[-1], 1000)
d10 = d1(x); d20 = d2(x) # 计算导数的具体值
L = np.trapz(np.sqrt(1 + d10 ** 2) + np.sqrt(1 + d20 ** 2), x) # 计算周长
L = L / 18 * 40 # 比例尺换算
print('周长L=', round(L, 4))
S = np.trapz(f2(x)-f1(x), x) # 计算面积
S = S / 18 ** 2 * 1600 # 比例尺换算
print('面积S=', round(S, 4))
delta = (S - 41288) / 41288 # 相对误差
print("相对误差delta=", round(delta, 4))
plt.show()