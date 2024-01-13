import numpy as np
from question3 import solve_all
from sko.GA import GA
import matplotlib.pyplot as plt
import pandas as pd
from sko.tools import set_run_mode

def ave_power(x):
    """定义目标函数"""
    b0, B0 = x
    x1 = np.array([-2])
    v1 = np.array([0])
    x2 = np.array([-1.8])
    v2 = np.array([0])
    theta1 = np.array([0])
    omega1 = np.array([0])
    theta2= np.array([0])
    omega2 = np.array([0])
    x1, v1, x2, v2, theta1, omega1, theta2, omega2 = solve_all(x1, v1, x2, v2, theta1, omega1, theta2, omega2, m3=1091.099, J3=7142.493, omega=1.9806, f=1760, L=2140, b1=528.5018, B1=1655.909, b0=b0, B0=B0)
    y1 = b0 * np.abs(v1 - v2) ** 2
    y2 = B0 * np.abs(omega1 - omega2) ** 2
    all_power = 0
    for i in range(len(y1) - 2001, len(y1) - 1):
        all_power += (y1[i] + y1[i+1]) * 0.01 / 2 + (y2[i] + y2[i+1]) * 0.01 / 2
    ave_power = all_power / (2000 * 0.01)
    return -ave_power # GA默认求最小值，加负号

set_run_mode(ave_power, 'parallel')
ga = GA(func=ave_power, n_dim=2, size_pop=200, max_iter=20, prob_mut=0.001, lb=[55000, 0], ub=[65000, 100000], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
