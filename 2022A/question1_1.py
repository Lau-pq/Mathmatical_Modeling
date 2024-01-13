import numpy as np
from scipy.integrate import solve_ivp
from numpy import fft
import pylab as plt
x0 = 0.5
k = 80000
m1 = 4866
m2 = 2433
b = 10000
f = 6250
omega = 1.4005
g = 9.8
pho = 1025
r = 1
V0 = 1/3 * np.pi * r * r * 0.8

def buoyancy(h) -> float:
    '''根据浮子距离海平面位置, 返回相应浮力'''
    if h < 0:
        return 0
    elif h < 0.8:
        r =  h / 0.8
        V = 1/3 * np.pi * r * r * h
    elif 0.8 <= h < 3.8:
        r = 1
        V = V0 + np.pi * r * r * (h - 0.8)
    else:
        r = 1
        V = V0 + np.pi * r * r * 3
    return pho * g * V

def solve(b, m3, f, omega, b1):
    A = np.array([[0, 1, 0, 0], 
                [-(k+pho * g * np.pi)/(m1+m3), -(b + b1)/(m1+m3), k/(m1+m3), b/(m1+m3)], 
                [0, 0, 0, 1], 
                [k/m2, b/m2, -k/m2, -b/m2]])
    B = np.array([[0, 0], 
                [1/(m1+m3), 0], 
                [0, 0],
                [0, 1/m2]])

    def func(t, x):
        u = np.array([f*np.cos(omega*t) - m1 * g + pho * g * V0 - k * x0, - m2 * g + k * x0])
        return [np.dot(A[i, :], x) + np.dot(B[i, :], u) for i in range(4)]

    N = (2 * np.pi / omega) * 40 + 0.2
    t_span = (0, N)
    t_eval = np.arange(0, N, 0.01)
    y0 = [-2.0, 0, -1.8, 0]

    sol = solve_ivp(func, t_span, y0, t_eval=t_eval)
    return sol
    
if __name__ == '__main__':
    sol = solve(b=10000, m3=1335.535, f=6250, omega=1.4005, b1=656.3616)


    plt.subplot(411)
    plt.plot(sol.t, sol.y.T[:, 0])
    plt.subplot(412)
    plt.plot(sol.t, sol.y.T[:, 1])
    plt.subplot(413)
    plt.plot(sol.t, sol.y.T[:, 2])
    plt.subplot(414)
    plt.plot(sol.t, sol.y.T[:, 3])
    print(sol.y.T[1000, 0] + 2, sol.y.T[1000, 1], sol.y.T[1000, 2] + 1.8, sol.y.T[1000, 3])
    print(sol.y.T[2000, 0] + 2, sol.y.T[2000, 1], sol.y.T[2000, 2] + 1.8, sol.y.T[2000, 3])
    print(sol.y.T[4000, 0] + 2, sol.y.T[4000, 1], sol.y.T[4000, 2] + 1.8, sol.y.T[4000, 3])
    print(sol.y.T[6000, 0] + 2, sol.y.T[6000, 1], sol.y.T[6000, 2] + 1.8, sol.y.T[6000, 3])
    print(sol.y.T[10000, 0] + 2, sol.y.T[10000, 1], sol.y.T[10000, 2] + 1.8, sol.y.T[10000, 3])
    plt.show()



