# 求方程在1.5附近的一个实根
from scipy.optimize import fsolve, root

fx = lambda x: x ** 980 - 5.01 * x ** 979 + 7.398 * x ** 978\
    - 3.388 * x ** 977 - x **3 + 5.01 * x ** 2 - 7.398 * x + 3.388
x1 = fsolve(fx, 1.5, maxfev=4000) # 函数调用4000次
x2 = root(fx, 1.5)
print(x1,'\n', '----------------------------')
print(x2)

# 求方程的数值解
from scipy.optimize import fsolve, root

fx = lambda x:[x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1]]
s1 = fsolve(fx, [1, 1])
s2 = root(fx, [1, 1])
print(s1,'\n','-------------------')
print(s2)