import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def creat_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)

"""数据提取、整合"""
# 去除第一列和最后一列数据
a = list(range(13))
a.remove(0)
# a.remove(10)
dataset = pd.read_csv('p_data.csv',usecols=a,encoding = 'utf-8', header=0)
dataset = dataset[5000:6000]
# print(dataset.values[1])
"""数据切片"""
number_train = 700
number_test = 300
n_day = 5
n_features = 12

values = dataset.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
scaled = scaler.fit_transform(values)
data = scaled
# 获取X,Y输入输出数据,将数据分割
X,Y =creat_dataset(data)
train_x = X[:number_train, :]
train_y = Y[:number_train, :]
test_x = X[number_train:number_train+number_test, :]
test_y = Y[number_train:number_train+number_test, :]

s = train_x
print(s,s.shape,'........')


# 定义DDPG模型
class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    # 构建Actor模型
    def build_actor(self):
        state_input = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        output = Dense(self.action_dim, activation='tanh')(x)
        output = output * self.action_bound
        model = Model(inputs=state_input, outputs=output)
        return model

    # 构建Critic模型
    def build_critic(self):
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        x = Dense(64, activation='relu')(state_input)
        x = tf.concat([x, action_input], axis=-1)
        x = Dense(64, activation='relu')(x)
        output = Dense(1)(x)
        model = Model(inputs=[state_input, action_input], outputs=output)
        return model

    # 选择动作
    def choose_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        print(state.shape)
        # 乘以[0]的目的是为了降维
        action = self.actor.predict(state)[0]
        return action

# 创建DDPG模型实例
state_dim = 60  # 状态维度
action_dim = 1  # 动作维度
action_bound = (0,1)  # 动作范围
ddpg = DDPG(state_dim, action_dim, action_bound)

# 使用DDPG模型进行时序预测
state = np.random.rand(state_dim)  # 输入状态
action = ddpg.choose_action(state)  # 选择动作

print("Predicted action:", action, action.shape)