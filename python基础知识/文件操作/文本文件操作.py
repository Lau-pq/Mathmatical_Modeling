# 遍历文件 data2_2.txt 中的所有行，统计每一行中字符的个数

with open('data2_2.txt') as fp:
    L1 = []; L2 = []
    for line in fp:
        L1.append(len(line))
        L2.append(len(line.strip())) # 去掉换行符
    data = [str(num) + '\t' for num in L2]
    print(L1); print(L2)
    with open('data2_42.txt', 'w') as fp2:
        fp2.writelines(data)

# 随机产生一个数据矩阵， 把它存入具有不同分隔符格式的文本文件中，再把数据从文本文件中提取出来

import numpy as np
a = np.random.rand(6, 8) # 生成 6*8 的 [0,1) 上均匀分布的随机数矩阵
np.savetxt("data2_43_1.txt", a) # 存成以制表符分隔的文本文件
np.savetxt('data2_43_2.csv', a, delimiter=',') # 存成以逗号分隔的CSV文件
b = np.loadtxt("data2_43_1.txt") # 加载空格分隔的文本文件
c = np.loadtxt("data2_43_2.csv", delimiter=',') # 加载CSV文件
print(b)