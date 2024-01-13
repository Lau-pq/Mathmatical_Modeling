import numpy as np
import pandas as pd
import pylab as plt

def generate_matrix(n,diag_a, diag_b, diag_c):
    '''使用对角矩阵相加得到三对角矩阵A'''
    a = np.zeros((n-1))
    b = np.zeros(n)
    array_b = np.insert(diag_b, (n-1), values=a, axis=0)# 添加行
    array_b = np.insert(array_b, 0, values=b, axis=1)# 添加列
    array_c = np.insert(diag_c, 0, values=a, axis=0)
    array_c = np.insert(array_c, (n-1), values=b, axis=1)
    matrix = diag_a + array_b + array_c
    return matrix


def matrix_ans(h, r, n, m, dx, u0):
    '''计算A B C d矩阵'''
    aa = np.array([1+h*dx])
    aa = np.append(aa, 2*np.ones(n-2)*(1+r))
    aa = np.append(aa, 1+h*dx)
    diag_aa = np.diag(aa)
    ab = np.array([-1])
    ab = np.append(ab, np.ones(n-2)*(-r))
    diag_ab = np.diag(ab)
    ac = np.ones(n-2)*(-r)
    ac = np.append(ac, -1)
    diag_ac = np.diag(ac)
    A = generate_matrix(n, diag_aa, diag_ab, diag_ac)
    ba = np.array([0])
    ba = np.append(ba,2*np.ones(n-2)*(1-r))
    ba = np.append(ba, 0)
    diag_ba = np.diag(ba)
    bb = np.array([0])
    bb = np.append(bb, np.ones(n-2)*r)
    diag_bb = np.diag(bb)
    bc = np.ones(n-2)*(r)
    bc = np.append(bc, 0)
    diag_bc = np.diag(bc)
    A = generate_matrix(n, diag_aa, diag_ab, diag_ac)
    B = generate_matrix(n, diag_ba, diag_bb, diag_bc)
    C = np.linalg.solve(A, B)# A @ C = B
    d = np.zeros((n, m))
    d[0,:] = h*dx*u0
    d[n-1,:] = d[0,:]
    d = np.linalg.solve(A,d)
    return C,d

def data_handle():
    '''对附件数据进行处理'''
    data = pd.read_excel('附件.xlsx').values
    x0 = data[:, 0]; y0 = data[:, 1]
    return x0, y0

def show_residuals(A:list):
    '''绘制残差图'''
    x0, y0 = data_handle()
    residuals = A[38:] - y0
    plt.title('模型温度与实验温度差')
    plt.xlabel('时间(t)')
    plt.ylabel('温度(℃)')
    plt.plot(x0, residuals)
    plt.show()

def plot_v(x, y, ylabel, xlabel='传送带过炉速度(cm/min)'):
    '''绘制传送带过炉速度与制程界限饿图'''
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


if __name__ == '__main__':
    x0, y0 = data_handle()
    plt.plot(x0, y0)
    plt.show()