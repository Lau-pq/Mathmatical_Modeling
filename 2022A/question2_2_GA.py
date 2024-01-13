from question1_2 import solve_all
import numpy as np
from sko.GA import GA
import matplotlib.pyplot as plt
import pandas as pd

def ave_power(x):
    """定义目标函数"""
    b, p = x
    x1 = np.array([-2])
    v1 = np.array([0])
    x2 = np.array([-1.8])
    v2 = np.array([0])
    x1, v1, x2, v2 = solve_all(x1, v1, x2, v2, m3=1165.992, omega=2.2143, f=4890, b0=b * 1e5, p=p, b1=167.8395)
    y = b * 1e5 * np.abs(v1 - v2) ** (2 + p)
    all_power = 0
    for i in range(len(y) - 2001, len(y) - 1):
        all_power += (y[i] + y[i+1]) * 0.01 / 2
    ave_power = all_power / (2000 * 0.01)
    return -ave_power # GA默认求最小值，加负号

ga = GA(func=ave_power, n_dim=2, size_pop=200, max_iter=20, prob_mut=0.001, lb=[0.35, 0], ub=[0.45, 0.05], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()