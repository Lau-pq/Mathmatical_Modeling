from compu_t import *
import pylab as plt
from env_temp import *
import csv

v = 78 / 60
t = compu_t(v=v, start_temp=[173, 198, 230, 257, 25])
time_total = warm_temp_x[11] / v
time = np.arange(0, time_total + 0.5, 0.5)

plt.rc('font', family='SimHei') # 用于正常显示中文标签 
plt.plot(time, t)
plt.title('炉温曲线')
plt.xlabel('时间(t)'); plt.ylabel('温度(℃)')
plt.show()

# # 小温区3 6 7 中点及 8结束处
print('小温区3中点:', t[171]) # x = 111.25 
print('小温区6中点:', t[335]) # x = 217.75
print('小温区7中点:', t[390]) # x = 253.25
print('小温区8结束处:', t[468]) # x = 304

# 写入 result.csv 文件
f = open('result.csv', 'w', newline='') # 创建文件对象
csv_writer = csv.writer(f) # 基于文件对象构建 csv 写入对象
csv_writer.writerow(["时间(s)","温度(摄氏度)"]) # 构建列表头
for i in range(len(t)):
    csv_writer.writerow([time[i], t[i]])
f.close()




