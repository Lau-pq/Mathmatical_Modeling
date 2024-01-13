import numpy as np
import matplotlib.pylab as plt
from question3 import depth


theta = np.pi / 3  # 半开角
alpha = 1.5 / 180 * np.pi  # 坡度
d = 200  # 测线间隔
eta = 0.1 # 最小覆盖率

plt.rc('font',family='SimHei')
plt.rc('axes', unicode_minus=False)

x = np.arange(-2, 2, 0.02) * 1852
y = - 110 + -x * np.tan(alpha)

plt.plot(x + 2 * 1852, y)
plt.xlabel('东西宽度(m)')
plt.ylabel('深度(m)')
plt.grid(True)


# beta = np.pi / 2 的情况
ans = np.array([])
max_x = 4 * 1852
x = 0
x2 = 0
while(x2 < max_x):
    y = depth(x)
    x1 = x
    x = x + np.abs(y) * np.tan(np.pi / 3)
    ans = np.append(ans, x)
    y = depth(x)
    W = np.abs(y) * np.sin(theta) * (1 / np.cos(theta - alpha) + 1 / np.cos(theta + alpha)) * np.cos(alpha) # 覆盖宽度
    # print(W / 1852)
    x2 = x1 + W
    x = x2 - eta * W

# print(ans)
# print(len(ans))
d = ans[1:] - ans[:-1]
print(d)
plt.figure()
plt.plot(depth(ans[:-1]), d)
plt.xlabel('深度(m)')
plt.ylabel('两条测线间隔(m)')
plt.title('深度和两条测线间隔的关系')
plt.show()
