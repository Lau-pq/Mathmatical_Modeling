from utils import *
import numpy as np
from compu_t import *

def compu_loss(t, start_t, end_t):
    '''损失函数: 残差平方和'''
    x0, y0 = data_handle()
    loss = t[start_t:end_t] - y0[start_t - 38:end_t - 38]
    L = np.sum(loss * loss)
    return L

def xm_upgrade(xm:list)->list:
    '''更新 xm'''
    return[6.767352754911945e-4, 0.736121979124664e4,
           7.6696482756524245e-4, 1.0594416951800402e4,
           9.362211057377521e-4, 1.0038676111023193e3,
           8.502179328929146e-4, 6.6995404368365065e2,
           xm[0], xm[1]] 
    

def F(alpha, h): # 定义目标函数
    alpha_bits = 1e-4; h_bits = 1e2
    xm = [alpha * alpha_bits, h * h_bits]
    xm = xm_upgrade(xm)
    t = compu_t(xm)
    L = compu_loss(t, 38, 747) #348, 409, 469, 592, 747
    return L


DNA_SIZE = 24 # DNA长度 二进制编码长度
POP_SIZE = 200 # 初始种群数量
N_GENERATIONS = 100  # 进化代数

X_BOUND = [1, 10]
Y_BOUND = [1, 10]

def translateDNA(pop):
    '''解码'''
    x_pop = pop[:, 1::2] # 奇数列表示x
    y_pop = pop[:, ::2] # 偶数列表示y
    # pop:(POP_SIZE * DNA_SIZE) * (DNA_SIZE, 1) --> (POP_SIZE, 1) 完成解码
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y

def get_fitness(pop):
    '''求最小值的适应度函数'''
    x, y = translateDNA(pop)
    pred = [0] * POP_SIZE
    for i in range(POP_SIZE):
        pred[i] = F(x[i], y[i])
    return -(pred - np.max(pred)) + 1e-3 # 防止适应度出现负值

def select(pop, fitness):
    '''自然选择, 适应度高的被选择机会多'''
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=(fitness)/(fitness.sum()))
    return pop[idx]

def crossover_and_mutation(pop, CROSSOVER_RATE=0.85):
    '''交叉、变异'''
    new_pop = []
    for father in pop: # 遍历种群中的每一个个体，将该个体作为父亲
        child = father # 孩子先得到父亲的全部基因
        if np.random.rand() < CROSSOVER_RATE: # 一定概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)] # 在种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2) # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:] # 孩子得到位于交叉点后母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.05):
    '''突变'''
    if np.random.rand() < MUTATION_RATE: # 以 MUTATION_RATE 的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 2) # 随机产生一个实数，代表要变异的基因位置
        child[mutate_point] = child[mutate_point] ^ 1 # 将变异点进行二进制反转

def print_info(pop):
    '''打印基因型'''
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print('max_fitness:', fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print('最优的基因型:', pop[max_fitness_index])
    print('(x, y):', (x[max_fitness_index], y[max_fitness_index]))


if __name__ == '__main__':
    # xm = [6.683140781153043e-04, 2.498727636628837e+04,
    # 8.076275611754757e-04, 1.431525270900793e+03,
    # 9.743913118484225e-04, 8.279199794106737e+02,
    # 8.492734595661612e-04, 6.547702252291799e+02, 
    # 5.286841991732021e-04, 1.337603876051590e+03]
    xm = [6.767352754911945e-4, 0.736121979124664e4, 
          7.6696482756524245e-4, 1.0594416951800402e4, 
          9.362211057377521e-4, 1.0038676111023193e3, 
          8.502179328929146e-4, 6.6995404368365065e2, 
          5.465401843214145e-4, 9.093668697099012e2]
    t = compu_t(xm)
    L = compu_loss(t, 38, 747)
    print(L)

    # pop 表示种群矩阵，一行表示一个二进制编码表示的DNA， 矩阵的行数为种群数目， DNA_SIZE为编码长度

    # pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
    # count = 0
    # for _ in range(N_GENERATIONS): # 种群迭代进化 N_GENERATIONS 代
    #     print(count)
    #     count+=1
    #     pop = np.array(crossover_and_mutation(pop)) # 种群通过交叉变异产生后代
    #     fitness = get_fitness(pop) # 对种群中每个个体进行评估
    #     pop = select(pop, fitness) # 选择产生新的种群
    # print_info(pop)
    

    
