# 三维曲线
import pylab as plt
import numpy as np

ax = plt.axes(projection='3d') # 设置三维图形模式
z = np.linspace(-50, 50, 1000)
x = z ** 2 * np.sin(z); y = z ** 2 * np.cos(z)
ax.plot(x, y, z, 'k')
plt.show()

# 三维表面图
import pylab as plt
import numpy as np
x = np.linspace(-4, 4, 100)
x, y = np.meshgrid(x, x) # 生成网格采样点
z = 50 * np.sin(x + y)
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, color='y')
plt.show()

ax = plt.axes(projection='3d')
X = np.arange(-6, 6, 0.25)
Y = np.arange(-6, 6, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
plt.colorbar(surf)
plt.show()


