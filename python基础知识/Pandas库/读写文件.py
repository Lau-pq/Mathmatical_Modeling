import pandas as pd
import numpy as np

# 生成服从正态分布的 24 * 4 的随机数矩阵，并保存为 DataFrame数据结构
dates = pd.date_range(start='20191101', end='20191124', freq='D')
a1 = pd.DataFrame(np.random.randn(24, 4), index=dates, columns=list('ABCD'))
a2 = pd.DataFrame(np.random.rand(24, 4))

# 数据写入文件
a1.to_excel('data2_38_1.xlsx')
a2.to_csv('data2_38_2.csv')
f = pd.ExcelWriter('data2_38_3.xlsx') # 创建文件对象
a1.to_excel(f, "Sheet1") # 把a1写入Excel文件
a2.to_excel(f, "Sheet2") # 把a2写入另一个表单中
f._save()

# 写入数据，不包含行索引
a1.to_excel('data2_38_4.xlsx', index=False) # 不包括行索引
a2.to_csv('data2_38_5.csv', index=False) # 不包括行索引
f = pd.ExcelWriter('data2_38_6.xlsx') # 创建文件对象
a1.to_excel(f, "Sheet1", index=False) # 把a1写入Excel文件
a2.to_excel(f, "Sheet2", index=False) # 把a2写入另一个表单中
f._save()

# 读入数据
a = pd.read_csv('data2_38_2.csv', usecols=range(1, 5))
b = pd.read_excel('data2_38_3.xlsx', "Sheet2", usecols=range(1, 5))
