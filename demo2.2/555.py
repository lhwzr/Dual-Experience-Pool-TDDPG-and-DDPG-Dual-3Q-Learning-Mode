# python 的学习测试
# 基础学习

'''
   1 让DDPG可以实现时间序列预测
   2 更改权值维度（与当前模型维度对应上）
   3 动作状态定义：状态为连续五个步长的12组指标；动作为一个输出的预测值
   !!! s定义为整体1000条数据，根据定义记忆库大小，使前30%为探索，后百分之七十为预测
'''


import gym
import numpy as np
import pandas as pd
from DDPG2 import DDPG
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# plt中文化处理
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def creat_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)
# 转置数据代码
def T_dataset(dataset):
    T_data = []
    for i in range(len(dataset)):
        a = np.transpose(dataset[i])
        T_data.append(a)
    return np.array(T_data)
"""数据提取、整合"""
# 去除第一列和最后一列数据
a = list(range(13))
a.remove(0)
# a.remove(10)
dataset = pd.read_csv('p_data.csv',usecols=a,encoding = 'utf-8', header=0)
dataset = dataset[5000:6000]
# print(dataset.values[1])
"""数据切片"""
number_train = 100
number_test = 300
n_day = 5
n_features = 12

values = dataset.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
scaled = scaler.fit_transform(values)
data = scaled
# 获取X,Y输入输出数据,将数据分割
X,Y = creat_dataset(data)
train_x = X[:number_train, :]
train_y = Y[:number_train, :]
test_x = X[number_train:number_train+number_test, :]
test_y = Y[number_train:number_train+number_test, :]
# !!!定义s_all为700组状态数据
s_all = train_x
r_data = pd.DataFrame(train_y)
print(train_y.shape)
print(train_y[0])
# s_all = T_dataset(s_all)
s_all = s_all.reshape(100,60)
# 定义一个数据长度，以便于确定一次强化学习的次数
data_len = len(train_x)  # 这里一个状态是五行12列的数组,一共700个
# # 循环次数200*50 = 10000 最大记忆库保存数
max_ep_step = 5
var = 3
"""手写环境及预测方式"""
# a_dim=12而不是a_dim=(1,12)是因为12个指标，每个输入一个动作，为一维12个数据
model = DDPG(a_dim=12, s_dim=60, a_bound=[0, 1],
             batch_size=32, tau=0.01, gamma=0.9, memory_capacity=1000)
for ep in range(max_ep_step):
    ep_reward = 0
    s = s_all[0]
    pre = []
    for i in range(data_len):
        # 根据状态做动作，（5×12组数据预测一组1×12的值）
        # 按照时间顺序从第一个状态开始取
        a = model.act(s)
        s_ = s_all[i+1]
        # 奖励值取均方差误差绝对值的倒数
        # r = -abs(mean_absolute_error(a,r_data.iloc[i, 0:12]))
        r = -abs(a-train_y[i])
        # print(r,"!!!!!!!")
        # 将当前的状态,行为,回报,下一个状态存储到记忆库中
        # print(s.shape,a.shape,r.shape)
        model.store_transition(s, a, r / 10, s_)

        # 达到记忆库容量的最大值
        if model.pointer > 100:
            var *= .9995  # 衰减动作随机性
            model.learn()  # 开始学习
            model.param_replace()  # 参数更新

        s = s_
        ep_reward += r
        i += 1
        pre.append(a)
        if i == data_len -1:
            print('Episode:', ep, ' Reward: %i', ep_reward, '预测值：', pre)

            pre = pd.DataFrame(pre)
            plt.plot(pre.iloc[:, 10:12])
            plt.plot(train_y[:, 10:12])
            # plt.plot(pre)
            plt.xlabel('Time')
            plt.ylabel('pre')
            plt.legend((u'r叶绿素a', u'r氟化物', u'p叶绿素a', u'p氟化物'), loc='upper right', fontsize='8')
            plt.title('lstm')
            # plt.grid()网格线设置
            plt.grid(True)
            plt.show()

            break

# for ep in range(100):
#     s = env.reset()  # 环境初始化 设置随机种子
#     ep_reward = 0
#     for step in range(max_ep_step):
#         if RENDER:
#             # 增加探索时的噪音
#             env.render()
#         # 先根据状态做出动作，在对该动作进行随即处理，增加前期探索性（用在时序预测时可以不用考虑，因为本身存在随机性，即每次训练时预测结果都不同）
#         a = model.act(s)
#         # 为行动选择添加随机性进行探索，action超过[-2,2]时做截断处理
#         a = np.clip(np.random.normal(a, var), -2, 2)
#         print(a)
#         s_, r, done, info = env.step(a)
#
#         # 将当前的状态,行为,回报,下一个状态存储到记忆库中
#         model.store_transition(s, a, r/10, s_)
#
#         # 达到记忆库容量的最大值
#         if model.pointer > 1000:
#             var *= .9995  # 衰减动作随机性
#             model.learn()  # 开始学习
#             model.param_replace()  # 参数更新
#
#         s = s_
#         print(s)
#         ep_reward += r
#         if step == max_ep_step -1:
#             print('Episode:', ep, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
#             if ep_reward > -300:  # 达到回合最大值且回合回报值大于-300,渲染环境
#                 RENDER = True
#             break
#
# env.close()  # 关闭渲染窗口