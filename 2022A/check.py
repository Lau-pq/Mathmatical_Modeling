import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import openpyxl

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

def pd_toexcel(data,filename): # pandas库储存数据到excel
    dfData = { # 用字典设置DataFrame所需数据
    '时间':data[0],
    '浮子位移':data[1],
    '浮子速度':data[2],
    '浮子角位移':data[3],
    '浮子角速度':data[4],
    '振子位移':data[5],
    '振子速度':data[6],
    '振子角位移':data[7],
    '振子角速度':data[8],
    }
    df = pd.DataFrame(dfData) # 创建DataFrame
    #用openpyxl打开excel
    wb=openpyxl.load_workbook(filename)
    #打开指定的Sheet
    ws = wb['Sheet1']
    
    startCol = 1
    startRow = 3
    
    #下面两行的意思是，将df1的每一行转成列表
    for i in range(0, df.shape[0]):
        eachRowList = df.iloc[i,:].tolist()
        #取每个列表里面的值
        for j in range(0,len(eachRowList)):
            #row 代表从几行开始， columns 代表从第几列开始
            ws.cell(row = startRow+i, column = startCol+j).value =eachRowList[j]
    
    #保存为新的表格
    wb.save(filename)

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

def a_p(v1, v2, omega1, omega2):
    b0=10000
    B0=1000
    y1 = b0 * np.abs(v1 - v2) ** 2
    y2 = B0 * np.abs(omega1 - omega2) ** 2
    all_power = 0
    for i in range(len(y1) - 2001, len(y1) - 1):
        all_power += (y1[i] + y1[i+1]) * 0.01 / 2 + (y2[i] + y2[i+1]) * 0.01 / 2
    ave_power_row = all_power / (2000 * 0.01)
    return ave_power_row

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
    p1 = a_p(v1, v2, omega1, omega2)
    print(p1)
    J1 = J1 * 0.99 # -1% 的扰动
    x11, v11, x21, v21, theta11, omega11, theta21, omega21 = solve_all(x1, v1, x2, v2, theta1, omega1, theta2, omega2, m3=1028.876, J3=7001.914, omega=1.7152, f=3640, L=1690, b1=683.4558, B1=654.3383 * 2, b0=10000, B0=1000)
    # J1 = J1 * 1.01 # +1% 的扰动
    # x11, v11, x21, v21, theta11, omega11, theta21, omega21 = solve_all(x1, v1, x2, v2, theta1, omega1, theta2, omega2, m3=1028.876, J3=7001.914, omega=1.7152, f=3640, L=1690, b1=683.4558, B1=654.3383 * 2, b0=10000, B0=1000)
    p2 = a_p(v11, v21, omega11, omega21)
    print(p2)
    print(abs(p1-p2)/p1)