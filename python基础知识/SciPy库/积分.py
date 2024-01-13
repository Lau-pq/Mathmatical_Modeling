from scipy.integrate import quad # 计算一重数值积分
def fun46(x, a, b):
    return a * x ** 2 + b * x
I1 = quad(fun46, 0, 1, args=(2, 1)) # 积分值 绝对误差
I2 = quad(fun46, 0, 1, args=(2, 10))
print(I1); print(I2)