# python 的学习测试
# 基础学习
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
'''

'''
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
        action = self.actor.predict(state)[0]
        return action

# 创建DDPG模型实例
state_dim = 10  # 状态维度
action_dim = 1  # 动作维度
action_bound = 1  # 动作范围
ddpg = DDPG(state_dim, action_dim, action_bound)

# 使用DDPG模型进行时序预测
state = np.random.rand(state_dim)  # 输入状态
action = ddpg.choose_action(state)  # 选择动作

print("Predicted action:", action)



import numpy as np

# 创建两个不同维度的数组
array1 = np.array([[1, 2, 3], [4, 5, 6]])  # 2行3列的数组
array2 = np.array([7, 8, 9])  # 1行3列的数组

# 使用np.vstack()函数垂直合并数组
result = np.vstack((array1, array2))

print(result)