import numpy as np
import matplotlib.pylab as plt

theta = np.pi / 3  # 半开角
alpha = 1.5 / 180 * np.pi  # 坡度
d = 200  # 测线间隔
eta = 0.1 # 最小覆盖率

def depth(x):
    """计算深度"""
    return - 110 - (x - 2 * 1852) * np.tan(alpha)


def dx(x, y, beta):
    """更新下一个测线最高点的坐标"""
    gamma = np.arcsin(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    z = depth(y)  # 深度
    W = np.abs(z) * np.sin(theta) * (1 / np.cos(theta - gamma) + 1 / np.cos(theta + gamma)) * np.cos(gamma) # 覆盖宽度
    N = np.abs(z) * np.sin(theta) * 1 / np.cos(theta + gamma) * np.cos(gamma) # 平分线一侧覆盖宽度
    y_1 = y + N * np.cos(beta - np.pi / 2) - W * eta
    L = N - W * eta + np.abs(depth(y_1)) * np.tan(np.pi / 3)
    if x < 2 * 1852:
        x_new = x + L / np.sin(beta - np.pi / 2) 
        y_new = y
        if x_new > 2 * 1852:
            y_new = (x_new - 2 * 1852) * np.tan(beta - np.pi / 2)
            x_new = 2 * 1852
    else :
        x_new = x
        y_new = y + L / np.cos(beta - np.pi / 2)
    return x_new, y_new


def init(beta):
    """初始化第一条测线坐标"""
    gamma = np.arctan(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    z = depth(0)  # 深度
    M = np.abs(z) * np.sin(theta) * 1 / np.cos(theta - gamma) * np.cos(gamma) # 平分线一侧覆盖宽度
    if beta == np.pi / 2:
        x = np.inf
    else:
        x = M / np.sin(beta - np.pi / 2)
    if x < 2 * 1852:
        return x, 0
    else:
        y = np.abs(z) * np.tan(np.pi / 3)
        return 2 * 1852, y
    
def far_y(y, beta):
    """测线侧量的最远点坐标"""
    gamma = np.arctan(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    z = depth(y)  # 深度
    N = np.abs(z) * np.sin(theta) * 1 / np.cos(theta + gamma) * np.cos(gamma) # 平分线一侧覆盖宽度
    return y + N * np.cos(beta - np.pi / 2)

def map(x, y, beta):
    """计算当前测线最低点坐标"""
    if x < 2 * 1852:
        return 0, np.tan(beta - np.pi / 2) * x
    else :
        return 0, y + 2 * 1852 * np.tan(beta - np.pi / 2)


def sum_len(beta):
    """计算测线总长度"""
    x, y = init(beta)
    sum = 0
    while x <= 2 * 1852 and far_y(y,beta) <= 4 * 1852:
        x_1, y_1 = map(x, y, beta)
        sum += np.sqrt((x_1 - x) ** 2 + (y_1 - y) ** 2)
        x, y = dx(x, y, beta)
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
        _, y_1 = map(x, y, beta)
        if y_f != y_0:
            d = (y_1 - y_f) * np.cos(beta - np.pi / 2)
            z_f = depth(y_f)
            W = np.abs(z_f) * np.sin(theta) * (1 / np.cos(theta - gamma) + 1 / np.cos(theta + gamma)) * np.cos(gamma) # 覆盖宽度
            eta = 1 - (d / W) * (np.cos(theta) * np.cos(gamma) / np.cos(theta - gamma))  # 覆盖率
            if eta > 0.2:
                flag = False
        y_f = y_1
        x, y = dx(x, y, beta)
    return flag


def show_line(beta):
    """绘制当前测线"""
    X = np.array([])
    Y = np.array([])
    x, y = init(beta)
    plt.plot((0, 4 * 1852),(0, 0), color = 'gray')
    plt.plot((4 * 1852, 4 * 1852),(0, 2 * 1852), color = 'gray')
    plt.plot((4 * 1852, 0),(2 * 1852, 2 * 1852), color = 'gray')
    plt.plot((0, 0),(2 * 1852, 0), color = 'gray')
    while x <= 2 * 1852 and far_y(y,beta) <= 4 * 1852:
        x_1, y_1 = map(x, y, beta)
        plt.plot((y,y_1),(x,x_1), color = 'red')
        # plt.plot((far_y(y,beta),far_y(y_1,beta)),(x,x_1), color = 'blue')
        X = np.append(X, x)
        Y = np.append(Y, y)
        x, y = dx(x, y, beta)
    x_1, y_1 = map(x, y, beta)
    plt.plot((y,y_1),(x,x_1), color = 'red')
    # plt.plot((far_y(y,beta),far_y(y_1,beta)),(x,x_1), color = 'blue')


if __name__ == '__main__':
    BETA = np.arange(90, 180, 0.01)
    # 求符合覆盖率的角度范围
    for beta in BETA:
        if(is_true(beta * np.pi / 180)):
            continue
        else:
            high_beta = beta
            break
    ans = np.array([])
    low = np.inf
    low_beta = 0
    print('最大可行角度为:' + str(high_beta) + '°')
    BETA = np.arange(90, high_beta - 0.01, 0.001)
    # 求最短测线路径
    for beta in BETA * np.pi / 180:
        ans_now = sum_len(beta)
        ans = np.append(ans, ans_now)
        if ans_now < low:
            low = ans_now
            low_beta = beta
    print('最佳测线方向夹角为:' + str(low_beta * 180 / np.pi) + '°')
    plt.rc('font',family='SimHei')
    plt.plot(BETA, ans)
    plt.xlabel('测线方向夹角 $\\beta$(°)')
    plt.ylabel('测线总长度(m)')
    plt.title('测线总长度与测线方向夹角关系')
    plt.figure()
    # 绘制测线路径
    show_line(low_beta)
    plt.xlabel('东西宽度(m)')
    plt.ylabel('南北宽度(m)')
    plt.title('测线路径')
    plt.show()

        


