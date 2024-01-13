import numpy as np
import pylab as plt

# 数据预处理
a = np.loadtxt('data12_1.txt')
x = a[:, 0]; y = a[:, 1]
d1 = np.array([70, 40]) # 起点
xy = np.vstack([d1, np.column_stack([x, y]), d1])
N = xy.shape[0] # N = 102
t = np.radians(xy) # 转化为弧度
R = 6370 # 地球半径
d = np.array([[R * np.arccos(np.cos(t[i, 0]-t[j, 0]) * np.cos(t[i, 1]) * np.cos(t[j, 1]) + np.sin(t[i, 1]) * np.sin(t[j, 1]))
               for j in range(N)] for i in range(N)]).real # 生成邻接矩阵

min_path = np.arange(N)
min_L = np.inf
min_seed = 0

for seed in range(10): # 4
    rng = np.random.default_rng(seed)

    # 初始化：Monte Carlo 法
    path = np.arange(N) # 保存路径 0, 1,..., 101
    L = np.inf # 保存最小值答案
    for _ in range(10000):
        path0 = np.arange(1, 101)
        rng.shuffle(path0) # 随机排序
        path0 = np.hstack([0, path0, 101]) # 打乱中间的 N - 2 项
        L0 = 0
        for i in range(101):
            L0 += d[path0[i], path0[i+1]] # 求总路程
        if L0 < L: # 更新
            path = path0
            L = L0
    # print(path, L, sep='\n')

    # 模拟退火
    e = 0.1 ** 30 # 终止温度
    M = 1000 # 没有使状态变化的循环次数上限
    alpha = 0.999 # 降温系数
    T = 100 # 起始温度
    cnt = 0
    while cnt < M:
        uv = rng.integers(1, 101, size=2)
        uv.sort()
        u, v = uv # 生成1, 2, ..., 100中的两个随机数, u < v 2 变换法
        df = (d[path[u-1], path[v]] + d[path[u], path[v+1]]
            - d[path[u-1], path[u]] - d[path[v], path[v+1]]) # 代价函数差
        if df < 0 or np.exp(-df/T) >= rng.random(1): # 接受准则
            path[u:v+1] = path[v:u-1:-1]
            L = L + df
            cnt = 0
        T *= alpha # 退火
        if T < e: 
            break
        cnt += 1
    # print(path, L, sep='\n')

    if L < min_L:
        min_L = L
        min_path = path
        min_seed = seed

print(min_seed, min_path, min_L, sep='\n')
xx = xy[path, 0]
yy = xy[path, 1]
plt.plot(xx, yy, '-o')
plt.show()  #画巡航路径


    





