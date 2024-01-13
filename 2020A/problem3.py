from optimize_5 import S
from compu_t import *
from pylab import plt

T1 = 172.6
T2 = 189.8
T3 = 225.1
T4 = 265.0
T5 = 25
v0 = 88.4
print('1~5小温区温度:', T1)
print('6小温区温度:', T2)
print('7小温区温度:', T3)
print('8~9小温区温度:', T4)
print('10·11小温区温度:', T5)
print('最小面积:', S(T1, T2, T3, T4, v0))
t = compu_t(v = v0 / 60, start_temp=[T1, T2, T3, T4, T5])
v = 88.4 / 60
dt = 0.5
t_total = warm_temp_x[11] / v 
m = floor (t_total / dt) + 1
x = np.arange(0 , m) * dt
time_max_index = np.argmax(t)
temp_rise_time = [t[time] for time in range(time_max_index+1) if t[time] >= 217]
rise_time = [time * dt for time in range(time_max_index+1) if t[time] >= 217]
plt.plot(x, t)
plt.fill_between(rise_time, 217, temp_rise_time, facecolor='blue', alpha=0.5)
plt.show()

