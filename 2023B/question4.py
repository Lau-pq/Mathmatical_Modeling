import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm

H = pd.read_excel(io='attach.xlsx',usecols='C:GU',header=1,names=None).values

def seabed_figure():
    """绘制海床图"""
    x = np.arange(0, 4 + 0.02, 0.02)
    y = np.arange(0, 5 + 0.02, 0.02)
    X , Y = np.meshgrid(x, y)

    plt.rc('font',family='SimHei')
    plt.rc('axes', unicode_minus=False)
    # 3D等高图
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, -H, cmap = 'hot', edgecolor='none')
    # cset = ax.contourf(X, Y, -H, zdir = 'z', offset = -200, cmap = cm.coolwarm)
    ax.set_xlabel('东西宽度(海里)')
    ax.set_ylabel('南北宽度(海里)')
    ax.set_zlabel('深度(m)')
    plt.title('海床图')
    # 平面等高图
    plt.figure()
    cset = plt.contourf(Y,X,-H,6,cmap='hot') 
    contour = plt.contour(Y,X,-H,20,colors='k')
    plt.clabel(contour,fontsize=10,colors='k')
    plt.colorbar(cset)
    plt.xlabel('南北宽度(海里)')
    plt.ylabel('东西宽度(海里)')
    plt.title('海床图')


def cal_gamma(x, y, b):
    """计算覆盖宽度线段与水平面夹角"""
    try :
        x1 = int((x - 100 * np.cos(b)) / (1852 * 0.02))
        x2 = int((x + 100 * np.cos(b)) / (1852 * 0.02))
        y1 = int((y - 100 * np.sin(b)) / (1852 * 0.02))
        y2 = int((y + 100 * np.sin(b)) / (1852 * 0.02))
        gamma = np.arctan((H[x1,y1] - H[x2,y2]) / 
                        (200 * 0.02 * 1852))
    except:
        gamma = 0
    return gamma


def cal_W(x, y, beta, h = 200 , theta = np.pi / 3, x0 = 5 * 1852, y0 = 4 * 1852):
    X = np.array([])
    Y = np.array([])
    W_s = np.array([])
    Z = np.array([])
    z = 0
    W_pr = 0
    sum = 0
    div = 0
    while(0 <= x <= x0 and 0 <= y <= y0 and z <= h):
        X = np.append(X, x)
        Y = np.append(Y, y)
        x_0 = int(x / (1852 * 0.02))
        y_0 = int(y / (1852 * 0.02))
        z = H[x_0, y_0]
        Z = np.append(Z, z)
        gamma = cal_gamma(x, y, np.pi - beta)
        W = z * np.sin(theta) * (1 / np.cos(theta - gamma) + 1 / np.cos(theta + gamma)) * np.cos(gamma)
        # print(W)
        W_s = np.append(W_s, W)
        if W_pr != 0:
            sum += (W_pr + W) * 100 / 2
            div += abs(W - W_pr)
        x += 100 * np.cos(beta - np.pi / 2)
        y += 100 * np.sin(beta - np.pi / 2)
        W_pr = W
    plt.plot(X / 1852, Y / 1852)
    return sum,div

def dx(x, y, beta, theta = np.pi / 3, eta = 0.1):
    """计算下一点坐标"""
    z = H[int(x / (1852 * 0.02)), int(y / (1852 * 0.02))]
    gamma = cal_gamma(x, y, np.pi - beta)
    W = z * np.sin(theta) * (1 / np.cos(theta - gamma) + 1 / np.cos(theta + gamma)) * np.cos(gamma)
    M = z * np.sin(theta) * 1 / np.cos(theta - gamma) * np.cos(gamma)   
    N = z * np.sin(theta) * 1 / np.cos(theta + gamma) * np.cos(gamma)
    if x == 0:
        L = N - W * eta
        x1 = x + L * np.sin(beta - np.pi / 2)
        y1 = y - L * np.cos(beta - np.pi / 2)
        z1 = H[int(x1 / (1852 * 0.02)), int(y1 / (1852 * 0.02))]
        L += z1 * np.tan(np.pi / 3)
        y = y - L / np.cos(beta - np.pi / 2)
        if y > 0:
            return x,y
        else:
            return np.abs(y) / np.tan(beta - np.pi / 2),0
    if y == 0:
        L = N - W * eta
        x1 = x + L * np.sin(beta - np.pi / 2)
        z1 = H[int(x1 / (1852 * 0.02)), 0]
        L += z1 * np.tan(np.pi / 3)
        x = x + L / np.sin(beta - np.pi / 2)
        return x,y
        
def init_beta(x,y):
    min_div = np.inf
    best_beta = 0
    for beta in np.arange(80, 120, 1) * np.pi / 180:
        _,div = cal_W(x, y, beta, x0 = np.cos(beta - np.pi / 2) * 200)
        if div < min_div:
            min_div = div
            best_beta = beta
    return best_beta

if __name__ == '__main__':
    # print(np.min(H), np.max(H))
    seabed_figure()
    # x, y = 0, 3.8 * 1852
    # sum_all = 0
    # best_beta = init_beta(x,y)
    # print(best_beta * 180 / np.pi)
    # while 0 <= x <= 5 * 1852 and 0 <= y <= 4 * 1852:
    #     sum,_ = cal_W(x,y, best_beta)
    #     sum_all += sum
    #     x, y = dx(x, y, best_beta)
    # # plt.figure()
    # # plt.plot(X, W)
    # # plt.figure()
    # # plt.plot(X, Z)
    # print(sum_all)
    plt.show()