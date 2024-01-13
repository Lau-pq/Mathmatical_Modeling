import numpy as np
from save_result import pd_toexcel1

theta = np.pi / 3  # 半开角
alpha = 1.5 / 180 * np.pi  # 坡度
d = 200  # 测线间隔

def cal(x, d):
    """计算深度、覆盖宽度、覆盖率"""
    D = 70 - d * x * np.tan(alpha)# 深度 
    W = D * np.sin(theta) * (1 / np.cos(theta - alpha) + 1 / np.cos(theta + alpha)) * np.cos(alpha) # 覆盖宽度
    eta = 1 - (d / W) * (np.cos(theta) * np.cos(alpha) / np.cos(theta - alpha))  # 覆盖率
    return D, W, eta

Data = np.zeros((9, 3))
for x in range(-4, 5):
    D, W, eta = cal(x, d)
    res = np.array([D, W, eta])
    Data[x + 4] = res

# 保存结果
pd_toexcel1(Data, 'result1.xlsx')
