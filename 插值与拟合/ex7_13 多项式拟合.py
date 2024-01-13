import numpy as np

a = np.loadtxt('data7_13.txt')
t = a[0]; y = a[1]
y = np.log(y) # 优化
p = np.polyfit(t, y, 1) # 拟合一次多项式
print("多项式的系数为:", p)
print("m=", round(p[0], 4))
print("k=", round(np.exp(p[1]), 4))