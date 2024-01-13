from scipy.integrate import odeint
import numpy as np
import pylab as plt
import sympy as sp

dy = lambda y, x: -2 * y + 2 * x ** 2 + 2 * x
xx = np.linspace(0, 3, 31)
s = odeint(dy, 1, xx)
print('x={}\n对应的数值解 y={}'.format(xx, s.flatten()))
plt.plot(xx, s, '*')
x = sp.var('x'); y = sp.Function('y')
eq = y(x).diff(x) + 2 * y(x) - 2 * x ** 2 - 2 * x
con = {y(0): 1}
s2 = sp.dsolve(eq, ics=con)
sx = sp.lambdify(x, s2.args[1], 'numpy') # 符号函数转匿名函数
plt.plot(xx, sx(xx))
plt.show()