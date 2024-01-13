import numpy as np
from scipy.integrate import solve_ivp
from question1_1 import pd_toexcel
import pylab as plt

x0 = 0.5
k = 80000
m1 = 4866
m2 = 2433
g = 9.8
rho = 1025
r = 1
V0 = 1/3 * np.pi * r * r * 0.8 
n = 0

def a_p(v1, v2):
    b0=10000
    y1 = b0 * np.abs(v1 - v2) ** 2
    all_power = 0
    for i in range(len(y1) - 2001, len(y1) - 1):
        all_power += (y1[i] + y1[i+1]) * 0.01 / 2
    ave_power_row = all_power / (2000 * 0.01)
    return ave_power_row


def solve_step(z:list, t_start, m3, omega, f, b0:float, p:float, b1:float)->list:
    dv = z[1] - z[3]
    b = (p+1) * b0 * abs(dv) ** p

    A = np.array([[0, 1, 0, 0], 
                  [-(k+rho * g * np.pi)/(m1+m3), -(b + b1)/(m1+m3), k/(m1+m3), b/(m1+m3)], 
                  [0, 0, 0, 1], 
                  [k/m2, b/m2, -k/m2, -b/m2]])
    B = np.array([[0, 0], 
                  [1/(m1+m3), 0], 
                  [0, 0], 
                  [0, 1/m2]])

    def func(t, x):
        u = np.array([f*np.cos(omega*t) - m1 * g + rho * g * V0 - (p * b0 * abs(dv) ** (p+1)) - k * x0, - m2 * g + (p * b0 * abs(dv) ** (p+1)) + k * x0])
        return [np.dot(A[i, :], x) + np.dot(B[i, :], u) for i in range(4)]

    dt = 0.01
    t_span = (t_start, t_start + 2 * dt)
    t_eval = np.arange(t_start, t_start + 2 * dt, 0.01)

    sol = solve_ivp(func, t_span, z, t_eval=t_eval)
    return sol.y.T[1]

def solve_all(x1, v1, x2, v2,x11, v11, x21, v21, lr, m3, omega, f, b0:float, p:float, b1:float):
    t = 0
    while t < (2 * np.pi / omega * 40 + 0.2):
        z = np.array([x1[-1], v1[-1], x2[-1], v2[-1]])
        z1 = np.array([x11[-1], v11[-1], x21[-1], v21[-1]])
        if t != 0 :
            lr += ((z - z1)/(np.abs(z) + np.ones(4)*0.01)) ** 2
        z = solve_step(z, t, m3, omega, f, b0, p, b1)
        z1 = solve_step(z1, t, m3, omega, f, b0, p, b1)
        t += 0.01
        x1 = np.append(x1,z[0])
        v1 = np.append(v1,z[1])
        x2 = np.append(x2,z[2])
        v2 = np.append(v2,z[3])
        x11 = np.append(x11,z1[0])
        v11 = np.append(v11,z1[1])
        x21 = np.append(x21,z1[2])
        v21 = np.append(v21,z1[3])
    return x1,v1,x2,v2,x11,v11,x21,v21

if __name__ == '__main__':
    omega = 1.4005
    z0 = [-2, 0, -1.8, 0]
    z01 = z0 + np.random.randn(4)
    print(z01)
    x1 = np.array([z0[0]])
    v1 = np.array([z0[1]])
    x2 = np.array([z0[2]])
    v2 = np.array([z0[3]])
    x11 = np.array([z01[0]])
    v11 = np.array([z01[1]])
    x21 = np.array([z01[2]])
    v21 = np.array([z01[3]])
    lr = np.zeros(4)
    x1, v1, x2, v2 ,x11, v11, x21, v21= solve_all(x1, v1, x2, v2,x11, v11, x21, v21, lr, m3=1335.535, f=6250, omega=1.4005, b0=10000, p=0.5, b1=656.3616)
    t = np.arange(0, (2 * np.pi / omega) * 40 + 0.2+0.01, 0.01)
    p1 = a_p(x1,x2)
    p2 = a_p(x11,x21)
    print(p1)
    print(p2)
    print(np.abs(p1-p2)/p1)

