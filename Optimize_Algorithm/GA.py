import numpy as np
import pylab as plt

def F(x, y):
	return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)- 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)- 1/3**np.exp(-(x+1)**2 - y**2)


DNA_SIZE = 24 # DNA长度 二进制编码长度
POP_SIZE = 200 # 初始种群数量
N_GENERATIONS = 1000  # 进化代数

X_BOUND = [-3, 3] # x 取值范围
Y_BOUND = [-3, 3] # y 取值范围

def translateDNA(pop):
    '''解码'''
    x_pop = pop[:, 1::2] # 奇数列表示x
    y_pop = pop[:, ::2] # 偶数列表示y
    # pop:(POP_SIZE * DNA_SIZE) * (DNA_SIZE, 1) --> (POP_SIZE, 1) 完成解码
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0] # 映射为 x 范围内的函数
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0] # 映射为 y 范围内的函数
    return x, y

def get_fitness(pop):
    '''求最大值的适应度函数'''
    x, y = translateDNA(pop)
    pred = [0] * POP_SIZE
    for i in range(POP_SIZE):
        pred[i] = F(x[i], y[i])
    return (pred - np.min(pred)) + 1e-3, pred # 防止适应度出现负值

# def get_fitness(pop):
#     '''求最小值的适应度函数'''
#     x, y = translateDNA(pop)
#     pred = [0] * POP_SIZE
#     for i in range(POP_SIZE):
#         pred[i] = F(x[i], y[i])
#     return -(pred - np.max(pred)) + 1e-3, pred # 防止适应度出现负值

def select(pop, fitness):
    '''自然选择, 适应度高的被选择机会多'''
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=(fitness)/(fitness.sum())) # 轮盘赌选择
    return pop[idx]

def crossover_and_mutation(pop, CROSSOVER_RATE=0.5):
    '''交叉、变异'''
    new_pop = []
    for father in pop: # 遍历种群中的每一个个体，将该个体作为父亲
        child = father # 孩子先得到父亲的全部基因
        if np.random.rand() < CROSSOVER_RATE: # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE*2) # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:] # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)
    return new_pop


def mutation(child, MUTATION_RATE=0.05):
    '''突变'''
    if np.random.rand() < MUTATION_RATE: # 以 MUTATION_RATE 的概率进行变异
        mutate_point = np.random.choice(np.arange(0, DNA_SIZE * 2), replace=False)
        child[mutate_point] = child[mutate_point] ^ 1 # 将变异点进行二进制反转

def print_info(pop):
    '''打印基因型'''
    fitness, pred = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    x, y = translateDNA(pop)
    print('最优的基因型:', pop[max_fitness_index])
    print('(x, y):', (x[max_fitness_index], y[max_fitness_index]))
    print('此时最优解:', F(x[max_fitness_index], y[max_fitness_index]))


if __name__ == '__main__':
    # pop 表示种群矩阵，一行表示一个二进制编码表示的DNA， 矩阵的行数为种群数目， DNA_SIZE为编码长度
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
    count = 0
    cur_best_std = [] ; cur_best_mean = []
    for _ in range(N_GENERATIONS): # 种群迭代进化 N_GENERATIONS 代
        print(count)
        count+=1
        pop = np.array(crossover_and_mutation(pop)) # 种群通过交叉变异产生后代
        fitness, pred = get_fitness(pop) # 对种群中每个个体进行评估
        cur_best_std.append(np.std(pred)); cur_best_mean.append(np.mean(pred))
        pop = select(pop, fitness) # 选择产生新的种群
        print_info(pop)
    plt.rc('font',family='SimHei')
    plt.rc('axes', unicode_minus=False)
    plt.title('遗传算法收敛过程示意图')
    plt.xlabel('进化次数(代)')
    plt.plot(range(N_GENERATIONS), cur_best_mean, c='violet')
    plt.plot(range(N_GENERATIONS), cur_best_std, c='gold')
    plt.legend(('种群适应度均值', '种群适应度标准差'))
    plt.grid()
    plt.show()
    

    
