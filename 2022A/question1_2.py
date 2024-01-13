import numpy as np
from scipy.integrate import solve_ivp
import pylab as plt

x0 = 0.5
k = 80000
m1 = 4866
m2 = 2433
g = 9.8
rho = 1025
r = 1
V0 = 1/3 * np.pi * r * r * 0.8 

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

def solve_all(x1, v1, x2, v2, m3, omega, f, b0:float, p:float, b1:float):
    t = 0
    while t < (2 * np.pi / omega * 40 + 0.2):
        z = [x1[-1], v1[-1], x2[-1], v2[-1]]
        z = solve_step(z, t, m3, omega, f, b0, p, b1)
        t += 0.01
        x1 = np.append(x1,z[0])
        v1 = np.append(v1,z[1])
        x2 = np.append(x2,z[2])
        v2 = np.append(v2,z[3])
    return x1,v1,x2,v2



if __name__ == '__main__':
    omega = 1.4005
    x1 = np.array([-2])
    v1 = np.array([0])
    x2 = np.array([-1.8])
    v2 = np.array([0])
    x1, v1, x2, v2 = solve_all(x1, v1, x2, v2, m3=1335.535, f=6250, omega=1.4005, b0=10000, p=0.5, b1=656.3616)
    t = np.arange(0, (2 * np.pi / omega) * 40 + 0.2+0.01, 0.01)
    plt.subplot(411)
    plt.plot(t, x1)
    plt.subplot(412)
    plt.plot(t, v1)
    plt.subplot(413)
    plt.plot(t, x2)
    plt.subplot(414)
    plt.plot(t, v2)
    print(x1[1000] + 2, v1[1000], x2[1000] + 1.8, v2[1000])
    print(x1[2000] + 2, v1[2000], x2[2000] + 1.8, v2[2000])
    print(x1[4000] + 2, v1[4000], x2[4000] + 1.8, v2[4000])
    print(x1[6000] + 2, v1[6000], x2[6000] + 1.8, v2[6000])
    print(x1[10000] + 2, v1[10000], x2[10000] + 1.8, v2[10000])
    plt.show()
    

