from scipy.interpolate import interp1d
import pylab as plt
import numpy as np

room_temp = 25 # 室温
# 每个大温区左右两端的温度
warm_temp_x = [0, 25, 197.5, 202.5, 233, 238, 268.5, 273.5, 339.5, 344.5, 410.5, 435.5]

def env_temp(start_temp:list=[175, 195, 235, 255, 25]):
    '''接受温区温度列表， 返回一个插值函数'''

    assert len(start_temp) == 5

    warm_temp = [None] * len(warm_temp_x)

    # 炉前区域（室温）
    warm_temp[0] = room_temp

    # 大温区1：小温区 1~5
    warm_temp[1] = start_temp[0]
    warm_temp[2] = start_temp[0]

    # 大温区2：小温区 6
    warm_temp[3] = start_temp[1]
    warm_temp[4] = start_temp[1]

    # 大温区3：小温区 7
    warm_temp[5] = start_temp[2]
    warm_temp[6] = start_temp[2]

    # 大温区4：小温区 8~9
    warm_temp[7] = start_temp[3]
    warm_temp[8] = start_temp[3]

    # 大温区5：小温区 10~11
    warm_temp[9] = start_temp[4]
    warm_temp[10] = start_temp[4]

    # 炉后区域（室温）
    warm_temp[11] = room_temp

    func = interp1d(warm_temp_x, warm_temp)
    return func



def show_env_temp(start_temp:list = [175, 195, 235, 255, 25])->None:
    '''调用env_temp并作环境温度的图'''
    
    func = env_temp(start_temp)
    x = np.arange(0, 436, 0.5)
    y = func(x)
    plt.plot(x, y)

if __name__ == "__main__":
    show_env_temp()
    plt.show()
    
    