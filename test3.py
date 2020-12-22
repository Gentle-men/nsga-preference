import numpy as np  # numpy库
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
raw_data = np.loadtxt('hhh.txt')  # 读取数据文件
x = raw_data[:, :-2]  # 分割自变量
x1 = raw_data[:, :1]  # 分割自变量
y1 = raw_data[:, -2]  # 分割因变量
y2 = raw_data[:, -1]  # 分割因变量
def minmax(min,max,x,e):
    for i in range(x.shape[0]):
        x[i][e]=(x[i][e]-min)/(max-min)
    return x
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


n_folds = 5  # 设置交叉检验的次数
model_br1 = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr1 = LinearRegression()  # 建立普通线性回归模型对象
# model_svr1 = SVR(kernel='rbf',C=1)  # 建立支持向量机回归模型对象
model_svr1 = SVR(gamma='scale', C=100)  # 建立支持向量机回归模型对象
model_ela1 = RandomForestRegressor(n_estimators=10, random_state=42)
model_br2 = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr2 = LinearRegression()  # 建立普通线性回归模型对象
model_svr2 = SVR(C=10, gamma=0.3)  # 建立支持向量机回归模型对象
model_ela2 = RandomForestRegressor(n_estimators=10, random_state=42)
model_names = ['BayesianRidge', 'LinearRegression', 'SVR', 'ElasticNet']  # 不同模型的名称列表
model_dic = [model_br1,model_lr1,model_svr1,model_ela1]  # 不同回归模型对象的集合
model_dic2 = [model_br2,model_lr2,model_svr2,model_ela2]  # 不同回归模型对象的集合
cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
pre_y_list2 = []  # 各个回归模型预测的y值列表

for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, x, y1, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(x, y1).predict(x))  # 将回归训练中得到的预测y存入列表
    pre_y_list2.append(model.fit(x, y2).predict(x))  # 将回归训练中得到的预测y存入列表
n_samples, n_features = x.shape  # 总样本量,总特征数
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
model_metrics_list2 = []  # 回归评估指标列表


for i in range(4):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    tmp_list2 = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(y1, pre_y_list[i])  # 计算每个回归指标结果
        tmp_score2 = m(y2, pre_y_list2[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
        tmp_list2.append(tmp_score2)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
    model_metrics_list2.append(tmp_list2)  # 将结果存入回归评估指标列表
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
df3 = pd.DataFrame(model_metrics_list2, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print(70 * '-')  # 打印分隔线
# print('cross validation result:')  # 打印输出标题
# print(df1)  # 打印输出交叉检验的数据框
# print(70 * '-')  # 打印分隔线
print('regression metrics:')  # 打印输出标题
print(df2)  # 打印输出回归指标的数据框
print(df3)  # 打印输出回归指标的数据框
print(70 * '-')  # 打印分隔线
print('short name \t full name')  # 打印输出缩写和全名标题
print('ev \t explained_variance')
print('mae \t mean_absolute_error')
print('mse \t mean_squared_error')
print('r2 \t r2')
print(70 * '-')  # 打印分隔线
plt.figure()  # 创建画布
plt.plot(np.arange(x.shape[0]), y1, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(x.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
plt.figure()  # 创建画布
plt.plot(np.arange(x.shape[0]), y2, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(x.shape[0]), pre_y_list2[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像


