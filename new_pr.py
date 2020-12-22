import numpy as np
import geatpy as ea
from sklearn.svm import SVR  # SVM中的回归算法
import pandas as pd
import csv
import time
XX=100
def minmax(min,max,x,e):
    for i in range(x.shape[0]):
        x[i][e]=(x[i][e]-min)/(max-min)
    return x
def re_minmax(min,max,x,e):
    m=x[e]*(max-min)
    x[e]=int(x[e]*(max-min)+min)
    return x
class MyProblem1(ea.Problem):
    def __init__(self):
        name='problem1'
        M=2
        maxormins=[1,1]
        Dim=4
        varTypes=[1,1,1,1]
        lb=[0,0,0,0]
        ub=[950,58,16,35]
        lbin = [1] * Dim  # 决策变量下边界 是否包含边界值，0表示不含边界值，1表示包含边界值
        ubin = [1] * Dim  # 决策变量上边界
        ea.Problem.__init__(self,name,M,maxormins,Dim,varTypes,lb,ub,lbin,ubin)
    def aimFunc(selfself,pop):
        Vars=pop.Phen
        gx1=Vars[:,0:4]
        gx2=Vars[:,:1]
        gx2=minmax(0,950,gx2,0)
        for i in range(len(gx1)):
            gx1[i][0] = gx2[i][0]
        gx2 = Vars[:, 1:2]
        gx2 = minmax(0, 58, gx2, 0)
        for i in range(len(gx1)):
            gx1[i][1] = gx2[i][0]
        gx2 = Vars[:, 2:3]
        gx2 = minmax(0, 16, gx2, 0)
        for i in range(len(gx1)):
            gx1[i][2] = gx2[i][0]
        gx2 = Vars[:, 3:4]
        gx2 = minmax(0, 35, gx2, 0)
        for i in range(len(gx1)):
            gx1[i][3] = gx2[i][0]
        # print(gx1)
        # print(gx1)
        ObjV1=10.5-m1.predict(gx1)
        ObjV2=m2.predict(gx1)
        # ObjV1 = m1.predict(gx1)
        # ObjV2 = m2.predict(gx1)
        pop.ObjV = np.array([ObjV1, ObjV2]).T
        # print(pop.ObjV)
class MyProblem2(ea.Problem):
    def __init__(self):
        name='problem2'
        M=2
        maxormins=[1,1]
        Dim=4
        varTypes=[1,1,1,1]
        lb=[0,0,0,0]
        ub=[950,58,16,35]
        lbin = [1] * Dim  # 决策变量下边界 是否包含边界值，0表示不含边界值，1表示包含边界值
        ubin = [1] * Dim  # 决策变量上边界
        ea.Problem.__init__(self,name,M,maxormins,Dim,varTypes,lb,ub,lbin,ubin)
    def aimFunc(selfself,pop):
        Vars=pop.Phen
        gx1=Vars[:,0:4]
        gx2=Vars[:,0:1]
        gx2=minmax(0,950,gx2,0)
        for i in range(len(gx1)):
            gx1[i][0] = gx2[i][0]
        ObjV1=-m1.predict(gx1)
        ObjV2=m2.predict(gx1)
        CV1 = 9-m1.predict(gx1)
        CV2 = m2.predict(gx1)-1.5
        pop.CV=np.array([CV1,CV2]).T
        pop.ObjV = np.array([ObjV1, ObjV2]).T
def getLoss(pre_list,y):
    loss=0
    for i in range(len(pre_list[0])):
        loss=loss+(pre_list[0][i]-y[i])*(pre_list[0][i]-y[i])
    return loss
def Cos_Theta(list_1,list_2):
    cos = 0
    A_modulus = 0
    B_modulus = 0
    if len(list_1)!=len(list_2):
        print('error')
        return 0
    else:
        for i in range(len(list_1)):
            cos+=list_1[i]*list_2[i]
            A_modulus+=list_1[i]*list_1[i]
            B_modulus+=list_2[i]*list_2[i]
    return cos/(np.sqrt(A_modulus)*np.sqrt(B_modulus))


if __name__=='__main__':
    import warnings
    warnings.filterwarnings("ignore")
    raw_data = np.loadtxt('hhh.txt')  # 读取数据文件
    x = raw_data[:, :-2]  # 分割自变量
    x1 = raw_data[:, :1]  # 分割自变量
    x2 = raw_data[:, 1:2]  # 分割自变量
    x3 = raw_data[:, 2:3]  # 分割自变量
    x4 = raw_data[:, 3:4]  # 分割自变量
    y1 = raw_data[:, -2]  # 分割因变量
    y2 = raw_data[:, -1]  # 分割因变量
    x1 = minmax(0, 950, x1, 0)
    for i in range(len(x)):
        x[i][0] = x1[i][0]
    x2 = minmax(0, 58, x2, 0)
    for i in range(len(x)):
        x[i][1] = x2[i][0]
    x3 = minmax(0, 16, x3, 0)
    for i in range(len(x)):
        x[i][2] = x3[i][0]
    x4 = minmax(0, 35, x4, 0)
    for i in range(len(x)):
        x[i][3] = x4[i][0]
    # print(x)
    m1 = SVR(gamma='scale', C=5).fit(x, y1)  # 将回归训练中得到的预测y存入列表
    m2 = SVR(C=10, gamma=0.3).fit(x, y2)  # 将回归训练中得到的预测y存入列表

    problem = MyProblem1()
    Encoding = 'RI'
    NIND = 50
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    myAl1 = ea.moea_NSGA2_templet(problem, population)
    myAl1.MAXGEN = 500
    myAl1.drawing = 0
    GD=[]
    IGD=[]
    GD_sum=0
    IGD_sum=0
    cos_sum=0
    clock=0
    count_sum=0
    for i in range(10):
        print(i)
        start = time.clock()
        NDSet2 = myAl1.run()
        end = time.clock()
        clock=clock+end-start
        NDSet2.save()
        PF = problem.getBest()  # 获取真实前沿
        if PF is not None and NDSet2.sizes != 0:
            GD1 = ea.indicator.GD(NDSet2.ObjV, PF)  # 计算GD指标
            IGD1 = ea.indicator.IGD(NDSet2.ObjV, PF)  # 计算IGD指标
        GD.append(GD1)
        IGD.append(IGD1)
        GD_sum+=GD1
        IGD_sum+=IGD1
        cos = 0
        count1 = 0
        normal = [2, 8]
        with open('./Result/ObjV.csv', 'r') as f:
            reader = csv.reader(f)
            row = list(reader)
            for i in row:
                i_float = map(float, i)
                i_float = list(i_float)
                tan_minus = 8 / 3
                tan_plus = 9 / 2
                tan_self = i_float[1] / i_float[0]
                if (tan_self < tan_plus and tan_minus < tan_self):
                    count1 += 1
                cos += Cos_Theta(i_float, normal)
            # print(len(row))
            cos = cos / len(row)
            # print(cos)
        count_sum+=count1
        cos_sum+=cos
    print(GD)
    print(IGD)
    print(cos_sum)
    print(count_sum)
    print(GD_sum/10)
    print(IGD_sum/10)
    print(cos_sum/10)
    print(clock/10)
    print(count_sum/10)

