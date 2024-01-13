import pandas as pd
import numpy as np

# 拆分、合并和分组计算
d = pd.DataFrame(np.random.randint(1, 6, (10, 4)), columns=list('ABCD'))
d1 = d[:4] # 获取前4行数据
d2 = d[4:] # 获取第5行以后的数据
dd = pd.concat([d1, d2]) # 数据行合并
s1 = d.groupby('A').mean() # 数据分组求均值
s2 = d.groupby('A').apply(sum) # 数据分组求和
print(d)
print(s1)
print(s2)

# 数据的选取与清洗
a = pd.DataFrame(np.random.randint(1, 6, (5, 3)), index=['a', 'b', 'c', 'd', 'e'], columns=['one', 'two', 'three'])
print(a)
a.loc['a', 'one'] = np.nan # 修改第1行第1列的数据
print(a)
b =  a.iloc[1:3, 0:2].values # 提取第2、3行、第1、2列数据
print(b)
a['four'] = 'bar'
print(a)
a2 = a.reindex(['a', 'b', 'c', 'd', 'e', 'f'])
print(a2)
a3 = a2.dropna() # 删除不确定值的行
print(a2)