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

for seed in range(10): # 7 
    rng = np.random.default_rng(seed)

    M = 50 # 种群大小
    G = 100 # 进化代数
    P = [] # 种群染色体矩阵

    # 改良圈算法生成初始种群
    for _ in range(M):
        c = np.arange(1, N-1)
        rng.shuffle(c)
        c = np.hstack([0, c, N-1]) # 染色体
        flag = 1
        while flag: # 更新直到不可更改
            flag = 0
            for u in np.arange(1, N-2):
                for v in np.arange(u+1, N-1):
                    if d[c[u-1], c[v]] + d[c[u], c[v+1]] < d[c[u-1], c[u]] + d[c[v], c[v+1]]:
                        c[u:v+1] = c[v:u-1:-1]
                        flag = 1
        c[c] = np.arange(N) # 编码
        P.append(c)
    P = np.asarray(P) / (N-1) # 归一化

    # 遗传算法
    for _ in range(G):
        A = P.copy()
        ord = np.arange(M) # 交叉的两两配对方式
        rng.shuffle(ord)
        pos = rng.integers(1, 100, M)
        for i in np.arange(0, M, 2): # 两两交叉
            A[ord[i], pos[i]:N-1], A[ord[i+1], pos[i]:N-1] = A[ord[i+1], pos[i]:N-1], A[ord[i], pos[i]:N-1]

        B = A.copy() # 变异
        by = []
        while len(by) < 1:
            by = np.where(rng.random(M) < 0.1)
        B = B[by]
        uvw = rng.integers(1, 101, size=3)
        uvw.sort()
        u, v, w = uvw 
        B = np.hstack([B[:, :u], B[:, v+1:w+1], B[:, u:v+1], B[:, w+1:]])

        K = np.vstack([P, A, B])
        ind1 = np.argsort(K, axis=1) # 下标排序，将染色体翻译为 0, 1,...,101
        NN = K.shape[0]
        LL = np.zeros(NN)
        for i in range(NN):
            for j in range(101):
                LL[i] += d[ind1[i, j], ind1[i, j+1]]
        ind2 = np.argsort(LL)
        P = K[ind2, :][:M, :] # 按适应程度排序选择出子染色体

    path = ind1[ind2[0], :]
    L = LL[ind2[0]]
    if L < min_L:
        min_L = L
        min_path = path
        min_seed = seed

print(min_seed, min_path, min_L, sep='\n')

xx = xy[path, 0]
yy = xy[path, 1]
plt.plot(xx, yy, '-o')
plt.show()  #画巡航路径