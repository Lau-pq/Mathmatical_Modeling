from compu_t import *
from env_temp import *
from utils import *
import pylab as plt

def temp_slope(t:list, dt:float=0.5)->float:
    '''温度斜率绝对值'''
    temp_slope = abs(np.diff(t) / dt)
    # print(np.max(temp_slope))
    return np.max(temp_slope)

def temp_rise_time(t:list, dt:float=0.5)->float:
    '''温度上升过程中在 150度 ~ 190度 的时间'''
    time_max_index = np.argmax(t)
    temp_rise_time = [t[time] for time in range(time_max_index+1) if t[time] >= 150 and t[time] <= 190]
    rise_time = (len(temp_rise_time) - 1) * dt
    # print(rise_time)
    return rise_time

def temp_greater_time(t:list, dt:float=0.5)->float:
    '''温度大于 217度 的时间'''
    temp_greater_time = [t[time] for time in range(len(t)) if t[time] > 217]
    greater_time = (len(temp_greater_time) - 1) * dt
    # print(greater_time)
    return greater_time

def temp_max(t:list)->float:
    '''最高温度'''
    t_max = np.max(t)
    # print(t_max)
    return t_max
    
def satisfy_process_boundaries(t:list)->bool:
    if 0 <= temp_slope(t) <= 3 and 60 <= temp_rise_time(t) <= 120 and 40 <= temp_greater_time(t) <= 90 and 240 <= temp_max(t) <= 250:
        return True
    else:
        return False

if __name__ == '__main__':
    # t = compu_t(v = 80 / 60, start_temp=[180, 190, 226, 263, 25])

    # print(satisfy_process_boundaries(t))

    target1 = [0] * 60
    target2 = [0] * 60
    target3 = [0] * 60
    target4 = [0] * 60

    for v in range(60, 120):
        t = compu_t(v=v / 60, start_temp=[182, 203, 237, 254, 25])
        target1[v-60] = temp_slope(t)
        target2[v-60] = temp_rise_time(t)
        target3[v-60] = temp_greater_time(t)
        target4[v-60] = temp_max(t)
    
    v = np.linspace(60, 120, 60)
    plt.rc('font',family='SimHei')
    plt.figure(figsize=(10,6))
    plt.subplot(221)
    plot_v(v, target1, '温度最大斜率')
    plt.subplot(222)
    plot_v(v, target2, '温度上升过程中150-190度时间(s)')
    plt.subplot(223)
    plot_v(v, target3, '温度大于217度的时间(s)')
    plt.subplot(224)
    plot_v(v, target4, '峰值温度(℃)')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig('figure_2', bbox_inches='tight')
    plt.show()




