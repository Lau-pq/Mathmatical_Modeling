import pylab as plt
import numpy as np
from scipy.interpolate import griddata

a = np.loadtxt('data7_9.txt')
x = a[0]; y = a[1]; z = -a[2]
xy = np.vstack([x, y]).T
xn = np.linspace(x.min(), x.max(), 100) # 插值点x的坐标
yn = np.linspace(y.min(), y.max(), 200) # 插值点y的坐标
xng, yng = np.meshgrid(xn, yn) # 构造网格节点
zn1 = griddata(xy, z, (xng, yng), method='cubic') # 三次样条插值
zn2 = griddata(xy, z, (xng, yng), method='nearest') # 最近邻插值
zn1[np.isnan(zn1)] = zn2[np.isnan(zn1)] # 把 nan 值替换
plt.rc('font', size=12); plt.rc('text', usetex=True)
ax = plt.subplot(121, projection='3d')
ax.plot_surface(xng, yng, zn1, cmap='viridis')
ax.set_xlabel('$x $'); ax.set_ylabel('$y $'); ax.set_zlabel('$z $')
plt.subplot(122)
c = plt.contour(xn, yn, zn1)
plt.clabel(c)
plt.subplots_adjust(wspace = 0.5) # 调整子图间距
plt.savefig('figure7_9.png', dpi=500)
plt.show()
