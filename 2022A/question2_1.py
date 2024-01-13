from question1_1 import solve
from scipy.integrate import solve_ivp
import numpy as np
import pylab as plt

x0 = 0.5
k = 80000
m1 = 4866
m2 = 2433
m3 = 1165.992
f = 4890
omega = 2.2143
g = 9.8
pho = 1025
r = 1
V0 = 1/3 * np.pi * r * r * 0.8

def ave_power(b):
    sol = solve(b, m3=1165.992, f=4890, omega=2.2143, b1=167.8395)
    y = b * (sol.y.T[:, 1] - sol.y.T[:, 3]) ** 2
    all_power = 0
    for i in range(len(y) - 2001, len(y) - 1):
        all_power += (y[i] + y[i+1]) * 0.01 / 2
    ave_power = all_power / (2000 * 0.01)
    return ave_power

if __name__ == '__main__':
    print(ave_power(37993))
    all_b = np.arange(0, 100000, 100)
    max = 0
    max_b = 0
    ave_power_list = []
    for b in all_b:
        ave_power_list.append(ave_power(b))
        print(b, ave_power(b))
        if(ave_power(b) > max):
            max = ave_power(b)
            max_b = b
    print('max:', max) # 229.99660485887816
    print("max_b", max_b) # 37993
    plt.rc('font', family='SimHei') # 用于正常显示中文标签 
    plt.plot(all_b, ave_power_list)
    plt.xlabel('直线阻尼系数')
    plt.ylabel('平均功率$P(w)$')
    plt.show()
        



