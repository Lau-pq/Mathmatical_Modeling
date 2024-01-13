import sympy as sp
# 定义符号变量或符号函数

x, y, z = sp.symbols('x, y, z') # 定义符号变量 x, y, z
f, g = sp.symbols('f, g', cls=sp.Function) # 定义多个符号函数
y = sp.Function('y') # 定义符号函数

sp.var('x, y, z') # 定义符号变量 x, y, z
sp.var('a b c') # 中间分隔符更换为空格
sp.var('f, g', cls=sp.Function) # 定义符号函数

# 求符号代数方程的解
a, b, c, x = sp.symbols('a, b, c, x')
x0 = sp.solve(a * x ** 2 + b * x + c, x)
print(x0)

# 求方程组的符号解
x1, x2 = sp.var('x1, x2')
s = sp.solve([x1 ** 2 + x2 ** 2 - 1, x1 - x2], [x1, x2])
print(s)

# 符号数组求方程组的符号解
x = sp.var('x:2') # 定义符号数组
s = sp.solve([x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1]], x)
print(s)

# 求特征值和特征向量的符号解
a = sp.Matrix([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
print("特征值为:", a.eigenvals())
print('特征向量:\n', a.eigenvects())
