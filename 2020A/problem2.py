from process_boundaries import *

def find_max_v(func, min_condition, max_condition, mode):
    '''二分法找最大的 v '''
    min_v = 70
    max_v = 85
    mid_v = (max_v + min_v) / 2
    while max_v - min_v > 1e-6:
        mid_v = (max_v + min_v) / 2
        t = compu_t(v=mid_v / 60, start_temp=[182, 203, 237, 254, 25])
        target = func(t)
        if (target > max_condition and mode == 'increase') or (target < min_condition and mode == 'decrease'):
            max_v = mid_v
        else:
            min_v = mid_v
    return mid_v

max_v = [0] * 4
max_v[0] = find_max_v(temp_slope, 0, 3, mode='increase')
max_v[1] = find_max_v(temp_rise_time, 60, 120, mode='decrease')
max_v[2] = find_max_v(temp_greater_time, 40, 90, mode='decrease')
max_v[3] = find_max_v(temp_max, 240, 250, mode='decrease') 
print(max_v)
best_max_v = min(max_v)
print(best_max_v)