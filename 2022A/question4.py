import numpy as np
from question3 import solve_all

def ave_power(b0, B0):
    """定义目标函数"""
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
    return ave_power

if __name__ == '__main__':
     print(ave_power(60448, 100000)) # 316.518