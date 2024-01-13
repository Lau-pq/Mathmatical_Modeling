from math import floor, ceil
from env_temp import *
from utils import *

def compu_t(xm=[6.767352754911945e-4, 0.736121979124664e4, 
          7.6696482756524245e-4, 1.0594416951800402e4, 
          9.362211057377521e-4, 1.0038676111023193e3, 
          8.502179328929146e-4, 6.6995404368365065e2, 
          5.465401843214145e-4, 9.093668697099012e2], v:float=70 / 60, start_temp:list=[175, 195, 235, 255, 25], l:float=0.015, dt:float=0.5, dx:float=1e-4):
    '''计算温度'''

    t_total = warm_temp_x[11] / v

    t1 = warm_temp_x[3] / v
    t2 = warm_temp_x[5] / v
    t3 = warm_temp_x[7] / v
    m = floor (t_total / dt) + 1
    m1 = floor(t1 / dt) + 1
    m2 = floor(t2 / dt) + 1
    m3 = floor(t3 / dt) + 1


    # 10个需优化的参数a，h
    # r = a^2*dt/dx^2
    r1 = xm[0] ** 2 * dt / (dx ** 2) 
    r2 = xm[2] ** 2 * dt / (dx ** 2)
    r3 = xm[4] ** 2 * dt / (dx ** 2)
    r4 = xm[6] ** 2 * dt / (dx ** 2)
    r5 = xm[8] ** 2 * dt / (dx ** 2)
    h1 = xm[1]; h2 = xm[3]; h3 = xm[5]; h4 = xm[7]; h5 = xm[9]
    n = ceil(l / dx) + 1
    u = np.zeros((n, m))
    u[:,0] = 25
    t = np.ones(m) * 25
    t0 = np.arange(0, m, 1) * dt
    u0 = np.array([])
    # u0 : 各时刻外界的环境温度
    func = env_temp(start_temp)
    for i in t0:
        if func(v*i) != None:
            u0 = np.append(u0, func(v*i))
        else:
            u0 = np.append(u0,25)
    k = ceil(l / (2 * dx))  # 测温点
    
    # 1-5小温区
    C1,d1 = matrix_ans(h1, r1, n, m, dx, u0)
    for j in range(0, m1-1):
        u[:,j+1] = C1@u[:,j] + d1[:,j+1]
        t[j+1] = u[k][j+1]

    # 6小温区
    C2,d2 = matrix_ans(h2, r2, n, m, dx, u0)
    for j in range(m1-1, m2-1):
        u[:,j+1] = C2@u[:,j] + d2[:,j+1]
        t[j+1] = u[k][j+1]

    # 7小温区
    C3,d3 = matrix_ans(h3, r3, n, m, dx, u0)
    for j in range(m2-1, m3-1):
        u[:,j+1] = C3@u[:,j] + d3[:,j+1]
        t[j+1] = u[k][j+1]

    # 8-9小温区
    C4,d4 = matrix_ans(h4, r4, n, m, dx, u0)
    # 10-11小温区
    C5,d5 = matrix_ans(h5, r5, n, m, dx, u0)
    for j in range(m3-1, m-1):
        if t[j] >= t[j-1]:
            u[:,j+1] = C4@u[:,j] + d4[:,j+1]
        else:
            u[:,j+1] = C5@u[:,j] + d5[:,j+1]
        t[j+1] = u[k][j+1]
    
    return t

if __name__ == '__main__':
    xm = [6.767352754911945e-4, 0.736121979124664e4, 
          7.6696482756524245e-4, 1.0594416951800402e4, 
          9.362211057377521e-4, 1.0038676111023193e3, 
          8.502179328929146e-4, 6.6995404368365065e2, 
          5.465401843214145e-4, 9.093668697099012e2]
    # xm = [6.683140781153043e-04, 2.498727636628837e+04,
    # 8.076275611754757e-04, 1.431525270900793e+03,
    # 9.743913118484225e-04, 8.279199794106737e+02,
    # 8.492734595661612e-04, 6.547702252291799e+02, 
    # 5.286841991732021e-04, 1.337603876051590e+03]
    
    v = 70 / 60
    dt = 0.5
    t_total = warm_temp_x[11] / v 
    m = floor (t_total / dt) + 1
    t = compu_t(v = 70 / 60)
    x = np.arange(0 , m) * dt
    print(x, t)
    x0, y0 = data_handle()
    plt.rc('font', family='SimHei') # 用于正常显示中文标签 
    plt.figure(figsize=(14, 6))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(121)
    plt.plot(x0, y0, label='实验温度')
    plt.plot(x,t, label='模型温度')
    plt.xlabel('时间(t)')
    plt.ylabel('温度(℃)')
    plt.legend(loc='lower right')
    plt.title('实验温度与模型温度比较图')
    plt.subplot(122)
    show_residuals(t)
    plt.show()