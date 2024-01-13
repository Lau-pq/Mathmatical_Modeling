import numpy as np

# 矩阵的合并
a = np.arange(16).reshape(4, 4) # 生成4行4列的矩阵
b = np.floor(5 * np.random.random((2, 4)))
c = np.ceil(6 * np.random.random((4, 2)))
d = np.vstack([a, b]) # 上下合并矩阵
e = np.hstack([a, c]) # 左右合并矩阵

# 矩阵的分割
a = np.arange(16).reshape(4, 4) 
b = np.vsplit(a, 2) # 行分割
print('行分割:\n', b[0], '\n', b[1])
c = np.hsplit(a, 4) # 列分割
print('列分割:\n', c[0], '\n', c[1], '\n', c[2], '\n', c[3])

# 矩阵的求和
a = np.array([[0, 3, 4], [1, 6, 4]])
b = a.sum() # 使用方法，求矩阵所有元素的和
c1 = sum(a) # 使用内置函数，求矩阵逐列元素的和
c2 = np.sum(a, axis=0) # 使用函数，求矩阵逐列元素的和
c3 = np.sum(a, axis=0, keepdims=True) # 逐列求和
print(c2.shape, c3.shape) # c2是(3,)数组,c3是(1,3)数组

# 矩阵的逐个元素运算
a = np.array([[0, 3, 4], [1, 6, 4]])
b = np.array([[1, 2, 3], [2, 1, 4]])
c = a / b # 两个矩阵对应元素相除
d = np.array([2, 3, 2])
e = a * d # d先广播成与a同维数的矩阵，再逐个元素相乘 
f = np.array([[3], [2]])
g = a * f # f先广播成与a同维数的矩阵，再逐个元素相乘
h = a ** (1/2) # a的逐个元素的1/2次幂

# 矩阵乘法
a = np.ones(4)
b = np.arange(2, 10, 2)
c = a @ b # a作为行向量，b作为列向量
d = np.arange(16).reshape(4, 4)
f = a @ d # a作为行向量
g = d @ a # a作为列向量

# 矩阵运算与线性代数

# 范数计算(模长)
a = np.array([[0, 3, 4], [1, 6, 4]])
b = np.linalg.norm(a, axis=1) # 求行向量2范数
c = np.linalg.norm(a, axis=0) # 求列向量2范数
d = np.linalg.norm(a) # 求矩阵2范数
print('行向量2范数为:', np.round(b, 4))
print('列向量2范数为:', np.round(c, 4))
print('矩阵2范数为:', np.round(d, 4))

# 求解线性方程组的唯一解
a = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x1 = np.linalg.inv(a) @ b # 第一种解法 inv 求矩阵的逆矩阵
x2 = np.linalg.solve(a, b) # 第二种解法
print(x1); print(x2)

# 求超定线性方程组的最小二乘解
a = np.array([[3, 1], [1, 2], [1, 1]])
b = np.array([9, 8, 6])
x = np.linalg.pinv(a) @ b # pinv 矩阵的广义逆矩阵
print(np.round(x, 4))

# 求矩阵的特征值和特征向量
a = np.eye(4) # 四阶单位阵
b = np.rot90(a) # 将a旋转90度
c, d = np.linalg.eig(b) # eig 求矩阵的特征值和特征向量 
print('特征值为:', c)
print('特征向量为:\n', d)
