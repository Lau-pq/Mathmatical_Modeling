import numpy as np

p = np.array([[0.2, 0.8, 0], [0.8, 0, 0.2], [0.1, 0.3, 0.6]])
a = np.vstack([p.T - np.eye(3), np.ones((1, 3))]) # 构造方程组系数矩阵
b = np.hstack([np.zeros(3), 1]) # 构造方程组常数项列
x = np.linalg.pinv(a) @ b # 求线性方程组的数值解
print('方法1解为:', np.round(x, 4))

val, vec = np.linalg.eig(p.T)
s = vec[:, 1] / sum(vec[:, 1]) # 最大特征值(特征值 1)对应的特征向量归一化
print('方法2解为:', np.round(s, 4))
