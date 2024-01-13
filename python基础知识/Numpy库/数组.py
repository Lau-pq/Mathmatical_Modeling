import numpy as np

# 数组生成1
a1 = np.array([1, 2, 3, 4]) # 生成整形数组
a2 = a1.astype(float)
a3 = np.array([1, 2, 3, 4], dtype=float) # 浮点数
print(a1.dtype); print(a2.dtype); print(a3.dtype)
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.arange(1, 5) # 生成数组[1, 2, 3, 4]
d = np.linspace(1, 4, 4) # 生成数组[1, 2, 3, 4]
e = np.logspace(1, 3, 3, base=2) # 生成数组[2, 4, 8]

# 数组生成2
a = np.ones(4, dtype=int) # 输出[1, 1, 1, 1]
b = np.ones((4,), dtype=int) # 同f
c = np.ones((4, 1)) # 四行一列浮点数组
d = np.zeros(4) # 输出[0, 0, 0, 0]
e = np.empty(3) # 生成3个元素的空数组行向量
f = np.eye(3) # 生成三阶单位阵
g = np.eye(3, k=1) # 第k对角线元素为1
h = np.zeros_like(a) # 生成与a同维数的全0数组

# 数组索引
a = np.arange(16).reshape(4, 4) # 生成4行4列的数组 0~15
b = a[1][2] # 输出6
c = a[1, 2] # 同b
d = a[1:2, 2:3] # 输出[[6]]
x = np.array([0, 1, 2, 1])
print(a[x == 1]) # 输出a的第2、4行的元素

