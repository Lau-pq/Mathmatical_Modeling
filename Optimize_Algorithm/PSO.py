import numpy as np
import pylab as plt

def F(X:list):  # 适应函数
    return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))

class Particle:
    '''初始化种群'''
    def __init__(self, max_x, max_v, dim):
        self.pos = [np.random.uniform(-max_x, max_x) for i in range(dim)] # 粒子的位置
        self.v = [np.random.uniform(-max_v, max_v) for i in range(dim)] # 粒子的速度
        self.best_pos = [0.0 for i in range(dim)] # 粒子最好的位置
        self.fitnessValue = F(self.pos) # 适应度函数值

    def set_pos(self, value):
        '''设置当前位置'''
        self.pos = value

    def get_pos(self):
        '''获取当前位置'''
        return self.pos
    
    def set_best_pos(self, value):
        '''设置最好位置'''
        self.best_pos = value

    def get_best_pos(self):
        '''获取最好位置'''
        return self.best_pos

    def set_v(self, value):
        '''设置粒子速度'''
        self.v = value

    def get_v(self):
        '''获取粒子速度'''
        return self.v
    
    def set_fitness_value(self, value):
        '''设置适应度值'''
        self.fitnessValue = value

    def get_fitness_value(self):
        '''获取适应度值'''
        return self.fitnessValue
        
class PSO:
    def __init__(self, dim, size, iter_num, max_x, max_v , best_fitness_value=float('Inf'), C1=2, C2=2, w=1):
        self.C1 = C1 # 个体学习因子
        self.C2 = C2 # 群体学习因子
        self.w = w # 惯性权重
        self.dim = dim # 粒子的纬度 —— 自变量的个数
        self.size = size # 粒子的个数
        self.iter_num = iter_num # 迭代的次数
        self.max_x = max_x # 粒子最远位置
        self.max_v = max_v # 粒子最大速度
        self.tol = tol # 截至条件
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)] # 种群最优位置
        self.fitness_val_list = [] # 每次迭代最优适应值

        # 对种群进行初始化
        self.Particle_list = [Particle(self.max_x, self.max_v, self.dim) for i in range(self.size)]

    def set_bestFitnessValue(self, value):
        '''设置最优适应度值'''
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        '''获取最优适应度值'''
        return self.best_fitness_value
    
    def set_bestPosition(self, i, value):
        '''设置最优位置'''
        self.best_position[i] = value

    def get_bestPosition(self):
        '''获取最优位置'''
        return self.best_position

    def updata_v(self, part:Particle):
        '''更新速度'''
        v_value = self.w * part.get_v() + self.C1 * np.random.rand() * (part.get_best_pos() - part.get_pos()) \
                + self.C2 * np.random.rand() * (self.get_bestPosition() - part.get_pos())
        if v_value > self.max_v:
            v_value = self.max_v
        elif v_value < -self.max_v:
            v_value = -self.max_v
        part.set_v(v_value)

    def update_pos(self, part:Particle):
        '''更新粒子位置'''
        pos_value = part.get_pos() + part.get_v()
        part.set_pos(pos_value)
        value = F(part.get_pos())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value < self.best_fitness_value():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)
    
    def update(self):
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.updata_v(part) # 更新速度
                self.update_pos(part) # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue()) # 每次迭代完把当前的最优适应度存到列表
            print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))
        return self.fitness_val_list, self.get_bestPosition()
    
if __name__ == '__main__':
    pso = PSO(2, 20, 1000, 1e6, 1)
    fit_vat_list, best_pos = pso.update()
    print('最优位置:' + str(best_pos))
    print('最优解:' + str(fit_vat_list[-1]))
    plt.plot(range(fit_vat_list), fit_vat_list, alpha=0.5)
            




    

