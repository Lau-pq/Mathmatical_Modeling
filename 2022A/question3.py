import numpy as np
from scipy.integrate import solve_ivp
import pylab as plt

x0 = 0.5 # 弹簧原长
k = 80000 # 直线弹簧刚度
m1 = 4866 # 浮子质量
m2 = 2433 # 振子质量
g = 9.8 # 重力加速度
rho = 1025 # 密度
r = 1 # 底面表面积
h = 0.5 # 振子高度
r1 = 0.5 # 振子半径
V0 = 1/3 * np.pi * r * r * 0.8 # 圆锥体积
L0 = 1.4 # 浮子质心到地面的距离
K1 = 250000 # 扭转弹簧刚度
K2 = 8890.7 # 静水力矩恢复系数
J1 = 8289.43

def solve_step(z1:list, z2:list, t_start, m3, J3, omega, f, L, b1:float, B1:float, b0:float, B0:float)->list:
    b = b0
    L1 = z1[2] - z1[0] + h / 2 # 振子质心到转轴距离
    J2 = m2 * (L1 ** 2) + m2 * (h ** 2 / 12 + r1 ** 2 / 4) # 振子的转动惯量

    A = np.array([[0, 1, 0, 0], 
                  [-(k+rho * g * np.pi)/(m1+m3), -(b + b1)/(m1+m3), k/(m1+m3), b/(m1+m3)], 
                  [0, 0, 0, 1], 
                  [k/m2, b/m2, -k/m2, -b/m2]])
    B = np.array([[0, 0], 
                  [1/(m1+m3), 0], 
                  [0, 0], 
                  [0, 1/m2]])
    
    C = np.array([[0, 1, 0, 0],
              [-(K1 + K2)/ (J1 + J3), -(B0 + B1) / (J1 + J3), K1 / (J1 + J3), B0 /(J1 + J3)],
              [0, 0, 0, 1],
              [K1 / J2, B0 / J2, (-K1)/ J2, -B0 / J2]])
    
    D = np.array([[0, 0], 
                [1/(J1+J3), 0], 
                [0, 0], 
                [0, 1/J2]])

    def func1(t, x):
        u1 = np.array([f*np.cos(omega*t) - m1 * g + rho * g * V0 - k * x0, - m2 * g + k * x0])
        return [np.dot(A[i, :], x) + np.dot(B[i, :], u1) for i in range(4)]
    
    def func2(t, x):
        u2 = np.array([L * np.cos(omega * t), 0])
        return [np.dot(C[i, :], x) + np.dot(D[i, :], u2) for i in range(4)]

    dt = 0.01
    t_span = (t_start, t_start + 2 * dt)
    t_eval = np.arange(t_start, t_start + 2 * dt, 0.01)

    sol1 = solve_ivp(func1, t_span, z1, t_eval=t_eval)
    sol2 = solve_ivp(func2, t_span, z2, t_eval=t_eval)
    return sol1.y.T[1], sol2.y.T[1]

def solve_all(x1, v1, x2, v2, theta1, omega1, theta2, omega2, m3, J3, omega, f, L, b1:float, B1:float, b0:float, B0:float):
    t = 0
    while t < (2 * np.pi / omega * 40 + 0.2):
        z1 = [x1[-1], v1[-1], x2[-1], v2[-1]]
        z2 = [theta1[-1], omega1[-1], theta2[-1], omega2[-1]]
        z1, z2 = solve_step(z1, z2, t, m3, J3, omega, f, L, b1, B1, b0, B0)
        t += 0.01
        x1 = np.append(x1,z1[0])
        v1 = np.append(v1,z1[1])
        x2 = np.append(x2,z1[2])
        v2 = np.append(v2,z1[3])
        theta1 = np.append(theta1, z2[0])
        omega1 = np.append(omega1, z2[1])
        theta2 = np.append(theta2, z2[2])
        omega2 = np.append(omega2, z2[3])
    return x1,v1,x2,v2,theta1, omega1, theta2, omega2

if __name__ == '__main__':
    omega = 1.7152
    x1 = np.array([-2])
    v1 = np.array([0])
    x2 = np.array([-1.8])
    v2 = np.array([0])
    theta1 = np.array([0])
    omega1 = np.array([0])
    theta2 = np.array([0])
    omega2 = np.array([0])
    x1, v1, x2, v2, theta1, omega1, theta2, omega2 = solve_all(x1, v1, x2, v2, theta1, omega1, theta2, omega2, m3=1028.876, J3=7001.914, omega=1.7152, f=3640, L=1690, b1=683.4558, B1=654.3383 * 2, b0=10000, B0=1000)
    t = np.arange(0, (2 * np.pi / omega) * 40 + 0.2 + 0.01, 0.01)
    print(x1[1000]+2, v1[1000], x2[1000]+1.8, v2[1000])
    print(x1[2000]+2, v1[2000], x2[2000]+1.8, v2[2000])
    print(x1[4000]+2, v1[4000], x2[4000]+1.8, v2[4000])
    print(x1[6000]+2, v1[6000], x2[6000]+1.8, v2[6000])
    print(x1[10000]+2, v1[10000], x2[10000]+1.8, v2[10000])
    print(theta1[1000], omega1[1000], theta2[1000], omega2[1000])
    print(theta1[2000], omega1[2000], theta2[2000], omega2[2000])
    print(theta1[4000], omega1[4000], theta2[4000], omega2[4000])
    print(theta1[6000], omega1[6000], theta2[6000], omega2[6000])
    print(theta1[10000], omega1[10000], theta2[10000], omega2[10000])
    plt.subplot(421)
    plt.plot(t, x1)
    plt.subplot(422)
    plt.plot(t, v1)
    plt.subplot(423)
    plt.plot(t, x2)
    plt.subplot(424)
    plt.plot(t, v2)
    plt.subplot(425)
    plt.plot(t, theta1)
    plt.subplot(426)
    plt.plot(t, omega1)
    plt.subplot(427)
    plt.plot(t, theta2)
    plt.subplot(428)
    plt.plot(t, omega2)
    plt.show()



