import sympy as sp

t = sp.var('t')
x1 = sp.var('x1', cls=sp.Function)
x2 = sp.var('x2', cls=sp.Function)
x3 = sp.var('x3', cls=sp.Function) # 定义三个符号函数
x = sp.Matrix([x1(t), x2(t), x3(t)]) # 列向量
A = sp.Matrix([[3, -1, 1], [2, 0, -1], [1, -1, 2]])
eq = x.diff(t) - A @ x
con = {x1(0): 1, x2(0): 1, x3(0): 1}
s = sp.dsolve(eq, ics=con)
print(s)

