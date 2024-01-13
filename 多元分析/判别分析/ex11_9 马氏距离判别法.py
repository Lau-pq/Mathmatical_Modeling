import numpy as np
import sympy as sp
from numpy.linalg import inv

f = open('data11_9.txt')
d = f.readlines()
a = [e.split() for e in d[:2]] # 提取 Af 字符串数据
a = np.array([list(map(eval, e)) for e in a])
b = [e.split() for e in d[2:]] # 提取 Apf 字符串数据
b = np.array([list(map(eval, e)) for e in b])
mu1 = a.mean(axis=1, keepdims=True); s1 = np.cov(a, ddof=1) # 均值和协方差矩阵
mu2 = b.mean(axis=1, keepdims=True); s2 = np.cov(b, ddof=1)
x1, x2 = sp.var('x1, x2')
X = sp.Matrix([x1, x2]) # X 为列向量
d1 = (X - mu1).T @ inv(s1) @ (X - mu1); d1 = sp.expand(d1)
d2 = (X - mu2).T @ inv(s2) @ (X - mu2); d2 = sp.expand(d2)
W = sp.lambdify('x1, x2', d1 - d2, 'numpy') # 判别函数
sol = W(np.array([1.24, 1.28, 1.40]), np.array([1.80, 1.84, 2.04]))
check1 =  W(a[0], a[1]); check2 = W(b[0], b[1])
print(check2)
print(np.round(sol, 4)) # 输出三个判别函数值