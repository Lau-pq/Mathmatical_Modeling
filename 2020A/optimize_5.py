from utils import *
import numpy as np
from compu_t import *
from process_boundaries import *
from env_temp import *

def S(T1, T2, T3, T4, v): # 定义目标函数
    '''求上升过程中大于217度的面积'''
    
    t = compu_t(v=v / 60, start_temp=[T1, T2, T3, T4, 25])
    if satisfy_process_boundaries(t):
        time_max_index = np.argmax(t)
        temp_list = [t[time] for time in range(time_max_index+1) if t[time] >= 217]
        sum = 0
        dt = 0.5
        for i in range(len(temp_list) - 1):
            sum += (temp_list[i] - 217 + temp_list[i+1] - 217) * dt / 2
        return sum
    else:
        return 1e10

def Sigma(T1, T2, T3, T4, v):
    t = compu_t(v=v / 60, start_temp=[T1, T2, T3, T4, 25]) 
    if satisfy_process_boundaries(t):
        dt = 0.5
        time_max_index = np.argmax(t)
        temp_rise_list = [t[time] for time in range(time_max_index+1) if t[time] >= 217]
        sum_rise = 0
        for i in range(len(temp_rise_list) - 1):
            sum_rise += (temp_rise_list[i] - 217 + temp_rise_list[i+1] - 217) * dt / 2
        time_rise = (len(temp_rise_list) - 1) * dt
        temp_drop_list = [t[time] for time in range(time_max_index, len(t)) if t[time] >= 217]
        sum_drop = 0
        for i in range(len(temp_drop_list) - 1):
            sum_drop += (temp_drop_list[i] - 217 + temp_drop_list[i+1] - 217) * dt / 2 
        time_drop = (len(temp_drop_list) - 1) * dt
        sigma = max(abs(sum_rise - sum_drop) / max(sum_rise, sum_drop), abs(time_rise - time_drop) / max(time_rise, time_drop))
        return sigma
    else:
        return 1e10

DNA_SIZE = 24 # DNA长度 二进制编码长度
POP_SIZE = 200 # 初始种群数量
N_GENERATIONS = 200  # 进化代数
A_BOUND = [165, 185]
B_BOUND = [185, 205]
C_BOUND = [225, 245]
D_BOUND = [245, 265]
E_BOUND = [65, 100]

def translateDNA(pop):
    '''解码'''
    a_pop = pop[:, 0::5]
    b_pop = pop[:, 1::5] 
    c_pop = pop[:, 2::5]
    d_pop = pop[:, 3::5]
    e_pop = pop[:, 4::5]
    # pop:(POP_SIZE * DNA_SIZE) * (DNA_SIZE, 1) --> (POP_SIZE, 1) 完成解码
    a = a_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (A_BOUND[1] - A_BOUND[0]) + A_BOUND[0]
    b = b_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (B_BOUND[1] - B_BOUND[0]) + B_BOUND[0]
    c = c_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (C_BOUND[1] - C_BOUND[0]) + C_BOUND[0]
    d = d_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (D_BOUND[1] - D_BOUND[0]) + D_BOUND[0]
    e = e_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (E_BOUND[1] - E_BOUND[0]) + E_BOUND[0]
    return a, b, c, d, e

def get_fitness(pop):
    '''求最小值的适应度函数'''
    a, b, c, d, e = translateDNA(pop)
    pred = [0] * POP_SIZE
    for i in range(POP_SIZE):
        pred[i] = 1e3 / Sigma(a[i], b[i], c[i], d[i], e[i])
    return pred # 防止适应度出现负值

def crossover_and_mutation(pop, CROSSOVER_RATE=0.9):
    '''交叉、变异'''
    new_pop = []
    for father in pop: # 遍历种群中的每一个个体，将该个体作为父亲
        child = father # 孩子先得到父亲的全部基因
        if np.random.rand() < CROSSOVER_RATE: # 一定概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)] # 在种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 5) # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:] # 孩子得到位于交叉点后母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.05):
    '''突变'''
    if np.random.rand() < MUTATION_RATE: # 以 MUTATION_RATE 的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 5) # 随机产生一个实数，代表要变异的基因位置
        child[mutate_point] = child[mutate_point] ^ 1 # 将变异点进行二进制反转

def select(pop, fitness):
    '''自然选择, 适应度高的被选择机会多'''
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=(fitness)/(np.sum(fitness)))
    return pop[idx]

def print_info(pop):
    '''打印基因型'''
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print('max_fitness:', fitness[max_fitness_index])
    a, b, c, d, e = translateDNA(pop)
    print('最优的基因型:', pop[max_fitness_index])
    print('(a, b, c, d, e):', (a[max_fitness_index], b[max_fitness_index], c[max_fitness_index], d[max_fitness_index], e[max_fitness_index]))
    print(Sigma(a[max_fitness_index], b[max_fitness_index], c[max_fitness_index], d[max_fitness_index], e[max_fitness_index]))


if __name__ == '__main__':

    # print(S(174.83467041460696, 187.21459401932918, 232.5391297065693, 262.59467587439275, 88.34539761217817))
    # print(S(180.80810283470765, 187.59173647116043, 230.55076036159755, 264.0908109599835, 90.7515156722972199))
    print(Sigma(175.5966788885998, 194.40765556142662, 225.0844860127262, 264.2522084267264, 86.49848023047926))
    print(Sigma(180.90822195459734, 187.64281765477762, 230.24380595945155, 264.73566053722266, 91.14182985674321))
    print(Sigma(172.6, 189.8, 225.1, 265.0, 88.4))
    print(S(172.6, 189.8, 225.1, 265.0, 88.4))
    # pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 5))
    # count = 0
    # for _ in range(N_GENERATIONS): # 种群迭代进化 N_GENERATIONS 代
    #     print(count)
    #     count+=1
    #     pop = np.array(crossover_and_mutation(pop)) # 种群通过交叉变异产生后代
    #     fitness = get_fitness(pop) # 对种群中每个个体进行评估
    #     pop = select(pop, fitness) # 选择产生新的种群
    # print_info(pop)
    