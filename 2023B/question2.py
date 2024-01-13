import numpy as np
from save_result import pd_toexcel2

theta = np.pi / 3  # 半开角
alpha = 1.5 / 180 * np.pi  # 坡度
D = 120  # 海域中心深度
d = 0.3 * 1852# 步长

def cal_w(x, beta):
    """计算深度、覆盖宽度"""
    gamma = np.arctan(np.sin(beta) * np.tan(alpha))  # 测线斜波交线和投影夹角
    tan_phi = -np.cos(beta) * np.tan(alpha)  # 爬升角
    D = 120 - d * x * tan_phi# 深度 
    W = D * np.sin(theta) * (1 / np.cos(theta - gamma) + 1 / np.cos(theta + gamma)) * np.cos(gamma) # 覆盖宽度
    return D, W

Data = np.zeros((8, 8))
for x in range(8):
    j = 0
    for b in np.arange(0 , 360 , 45) * np.pi / 180:
        _, W = cal_w(x, b)
        Data[x, j] = W
        j += 1

# 保存结果
pd_toexcel2(Data, 'result2.xlsx')