import numpy as np
import matplotlib.pylab as plt

theta = np.pi / 3  # 半开角
alpha = 1.5 / 180 * np.pi  # 坡度
d = 200  # 测线间隔
eta = 0.1

def depth(x):
    """计算深度"""
    return - 110 - (x - 2 * 1852) * np.tan(alpha)


def map(x, y, beta):
    """计算当前测线最低点坐标"""
    if x < 2 * 1852:
        return 0, np.tan(beta - np.pi / 2) * x
    else :
        return 0, y + 2 * 1852 * np.tan(beta - np.pi / 2)


def dx(x, y, beta):
    """更新下一个测线最高点的坐标"""
    gamma = np.arcsin(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    x_1, y_1 = map(x, y, beta)
    z = depth(y)  # 深度
    z_1 = depth(y_1)
    W = np.abs(z) * np.sin(theta) * (1 / np.cos(theta - gamma) + 1 / np.cos(theta + gamma)) * np.cos(gamma) # 覆盖宽度
    M = np.abs(z) * np.sin(theta) * 1 / np.cos(theta - gamma) * np.cos(gamma) # 平分线一侧覆盖宽度
    W_1 = np.abs(z_1) * np.sin(theta) * (1 / np.cos(theta - gamma) + 1 / np.cos(theta + gamma)) * np.cos(gamma) # 覆盖宽度
    d = (W + W_1) * 9 / 20
    y_2 = y + (d - M) * np.cos(beta - np.pi / 2)
    L = d - M + np.abs(depth(y_2)) * np.tan(np.pi / 3)
    if x < 2 * 1852:
        x_new = x + L / np.sin(beta - np.pi / 2) 
        y_new = y
        if x_new > 2 * 1852:
            # y_new = y + L / np.cos(beta - np.pi / 2)
            y_new = (x_new - 2 * 1852) * np.tan(beta - np.pi / 2)
            x_new = 2 * 1852
    else :
        x_new = x
        y_new = y + L / np.cos(beta - np.pi / 2)
    return x_new, y_new, (d - M) / np.tan(beta - np.pi / 2) 


def init(beta):
    """初始化第一条测线坐标"""
    gamma = np.arctan(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    z = depth(0)  # 深度
    M = np.abs(z) * np.sin(theta) * 1 / np.cos(theta - gamma) * np.cos(gamma) # 平分线一侧覆盖宽度
    x = M / np.sin(beta - np.pi / 2)
    if x < 2 * 1852:
        return x, 0
    else:
        y = np.abs(z) * np.tan(np.pi / 3)
        return 2 * 1852, y
    
def far_y(y, beta):
    gamma = np.arctan(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    z = depth(y)  # 深度
    N = np.abs(z) * np.sin(theta) * 1 / np.cos(theta + gamma) * np.cos(gamma) # 平分线一侧覆盖宽度
    return y + N * np.cos(beta - np.pi / 2)

def low_y(y, beta):
    gamma = np.arctan(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    z = depth(y)  # 深度
    M = np.abs(z) * np.sin(theta) * 1 / np.cos(theta - gamma) * np.cos(gamma) # 平分线一侧覆盖宽度
    return y - M * np.cos(beta - np.pi / 2)


def sum_len(beta):
    """计算测线总长度"""
    x, y = init(beta)
    sum = 0
    _,_,d = dx(x, y, beta)
    while x <= 2 * 1852 and far_y(y,beta) <= 4 * 1852:
        if x < 2 * 1852:
            sum += d
        x_1, y_1 = map(x, y, beta)
        sum += np.sqrt((x_1 - x) ** 2 + (y_1 - y) ** 2)
        x, y, d = dx(x, y, beta)
    x_1, y_1 = map(x, y, beta)
    sum += np.sqrt((x_1 - x) ** 2 + (y_1 - y) ** 2)
    return sum


def is_true(beta):
    """判断是否是合格的测线"""
    gamma = np.arcsin(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    x, y = init(beta)
    _, y_0 = map(x, y, beta)
    y_f = y_0
    flag = True
    while x <= 2 * 1852 and y <= 4 * 1852:
        # if x < 2 * 1852:

        _, y_1 = map(x, y, beta)
        if y_f != y_0:
            d = (y_1 - y_f) * np.cos(beta - np.pi / 2)
            z_f = depth(y_f)
            W = np.abs(z_f) * np.sin(theta) * (1 / np.cos(theta - gamma) + 1 / np.cos(theta + gamma)) * np.cos(gamma) # 覆盖宽度
            eta = 1 - (d / W) * (np.cos(theta) * np.cos(gamma) / np.cos(theta - gamma))  # 覆盖率
            if eta > 0.2:
                flag = False
            # print(eta)
        y_f = y_1
        x, y, _ = dx(x, y, beta)
    return flag


def show_line(beta):
    """绘制当前测线"""
    X = np.array([])
    Y = np.array([])
    x, y = init(beta)
    # plt.plot((0, 4 * 1852),(0, 0), color = 'gray')
    # plt.plot((4 * 1852, 4 * 1852),(0, 2 * 1852), color = 'gray')
    # plt.plot((4 * 1852, 0),(2 * 1852, 2 * 1852), color = 'gray')
    plt.plot((0, 0),(2 * 1852, 0), color = 'gray')
    while x <= 2 * 1852 and far_y(y,beta) <= 0.3 * 1852:
        x_1, y_1 = map(x, y, beta)
        plt.plot((y,y_1),(x,x_1), color = 'red')
        plt.plot((far_y(y,beta),far_y(y_1,beta)),(x,x_1), color = 'blue')
        plt.plot((low_y(y,beta),low_y(y_1,beta)),(x,x_1), color = 'green')
        X = np.append(X, x)
        Y = np.append(Y, y)
        x, y, _ = dx(x, y, beta)
    x_1, y_1 = map(x, y, beta)
    plt.plot((y,y_1),(x,x_1), color = 'red')
    plt.plot((far_y(y,beta),far_y(y_1,beta)),(x,x_1), color = 'blue')
    plt.plot((low_y(y,beta),low_y(y_1,beta)),(x,x_1), color = 'green')


if __name__ == '__main__':
    BETA = np.arange(90, 180, 0.01)
    for beta in BETA:
        if(is_true(beta * np.pi / 180)):
            continue
        else:
            high_beta = beta
            break
    ans = np.array([])
    low = np.inf
    low_beta = 0
    BETA = np.arange(90, 91.8, 0.01)
    for beta in BETA * np.pi / 180:
        ans_now = sum_len(beta)
        ans = np.append(ans, ans_now)
        if ans_now < low:
            low = ans_now
            low_beta = beta
    print(low_beta * 180 / np.pi)
    plt.rc('font',family='SimHei')
    plt.plot(BETA, ans)
    plt.xlabel('测线方向夹角 $\\beta$ (°)')
    plt.ylabel('测线总长度(m)')
    plt.title('测线总长度与测线方向夹角关系')
    plt.figure()
    plt.grid(True)
    show_line(low_beta)
    plt.show()
