# python 的学习测试
# 基础学习
import tensorflow as tf
from tensorflow import keras
import numpy as np

class DDPG(keras.Model):
    def __init__(self, s_dim, a_dim, a_bound, batch_size=32, tau=0.002, gamma=0.95,
                 a_lr=0.0001, c_lr=0.001, memory_capacity=9000):
        super().__init__()
        self.batch_size = batch_size  # 批量数据
        self.tau = tau   # 滑动平均参数
        self.gamma = gamma   # 回报折扣系数
        self.a_lr = a_lr  # actor学习率
        self.c_lr = c_lr  # critic学习率
        self.memory_capacity = memory_capacity  # 记忆库大小
        # s_dim * 2 + a_dim + 1相当于一个记忆条存储当前状态、下一时刻状态、当前动作和当前奖励值
        self.memory = np.zeros((memory_capacity, s_dim * 2 + a_dim + 12), dtype=np.float32)
        # 记忆库初始化
        # print(self.memory,len(self.memory),self.memory.shape)
        self.pointer = 0  # 记忆库初始大小为0
        self.memory_full = False  # 记忆库是否已经满

        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound[1]  # 动作维度， 状态维度， 动作值的上限

        # s = keras.Input(shape=s_dim)     # current state
        # s_ = keras.Input(shape=s_dim)    # next state
        self.actor = self._build_actor(trainable=True, name="a/eval")  # 策略网络
        self.actor_ = self._build_actor(trainable=False, name="a/target")  # 目标策略网络
        self.actor_.set_weights(self.actor.get_weights())
        self.critic = self._build_critic(trainable=True, name="d/eval")  # Q网络
        self.critic_ = self._build_critic(trainable=False, name="d/target")  # 目标Q网络
        self.critic_.set_weights(self.critic.get_weights())
        self.a_opt = keras.optimizers.Adam(self.a_lr)  # 优化器
        self.c_opt = keras.optimizers.Adam(self.c_lr)  # 优化器
        self.mse = keras.losses.MeanSquaredError()  # 均方差损失函数

    def _build_actor(self, trainable, name):  # 设计策略网络
        data = keras.Input(shape=self.s_dim)
        # print('actor输入数据：', data, data.shape)
        x = keras.layers.Dense(30, activation="relu", trainable=trainable)(data)
        x = keras.layers.Dense(30, activation="relu", trainable=trainable)(x)
        # ！！！actor网络的输出是动作，所以维度和个数要对应
        x = keras.layers.Dense(12, trainable=trainable)(x)
        a = self.a_bound * tf.math.tanh(x)
        # print('actor输入数据：',a,a.shape)
        model = keras.Model(inputs=data, outputs=a, name=name)
        return model

    def _build_critic(self, trainable, name):  # 设计评价网络
        # ！！！critic网络输入时状态+动作，所以维度和个数要对应
        data = keras.Input(shape=(self.a_dim + self.s_dim,))
        # print('critic输入数据：', data, data.shape)
        x = keras.layers.Dense(30, activation="relu", trainable=trainable)(data)
        x = keras.layers.Dense(30, activation="relu", trainable=trainable)(x)
        q = keras.layers.Dense(12, trainable=trainable)(x)
        # print('Q表:', q, q.shape, '8')
        model = keras.Model(data, q, name=name)
        return model

    def param_replace(self):  # 参数更新
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        actor_target_weights = self.actor_.get_weights()
        critic_target_weights = self.critic_.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_target_weights[i] * (1 - self.tau) + self.tau * actor_weights[i]
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_target_weights[i] * (1 - self.tau) + self.tau * critic_weights[i]
        self.actor_.set_weights(actor_target_weights)
        self.critic_.set_weights(critic_target_weights)

    def act(self, s):  # 根据当前状态s执行动作
        a = self.actor.predict(np.reshape(s, (-1, self.s_dim)), verbose=0)[0]  # 使用策略网络
        return a

    def sample_memory(self):  # 从记忆库中采样数据
        indices = np.random.choice(self.memory_capacity, size = self.batch_size)
        bt = self.memory[indices, :]  # 获取批次数据
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 12: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        return bs, ba, br, bs_

    def learn(self):  # 训练策略网络和Q网络
        bs, ba, br, bs_ = self.sample_memory()
        with tf.GradientTape() as tape:  # 更新策略网络
            a = self.actor(bs)  # 获取策略网络执行的动作
            q = self.critic(tf.concat([bs, a], 1))
            actor_loss = tf.reduce_mean(-q)  # 最大化价值函数值Q等于最小化-Q
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)  # 仅更新策略网络参数
        self.a_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:  # 更新价值网络
            a_ = self.actor_(bs_)  # 目标策略网络根据下一状态决定下一动作
            q_ = br + self.gamma * self.critic_(tf.concat([bs_, a_], 1))
            q = self.critic(tf.concat([bs, ba], 1))
            critic_loss = self.mse(q_, q)  # 均方差损失函数
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)  # 仅更新价值网络参数
        self.c_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        return actor_loss.numpy(), critic_loss.numpy()

    # def store_transition(self, s, a, r, s_):  # 保存数据到记忆库中
    #     transition =  np.column_stack((s, a, [r], s_))
    #     index = self.pointer % self.memory_capacity
    #     self.memory[index, :] = transition
    #     self.pointer += 1

    def store_transition(self, s, a, r, s_):
        # transition = np.concatenate((s, a, r, s_))
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.memory_capacity
        # 每次存入一个12*12的数据（5*12+1*12+1*12+5*12）
        self.memory[index, :] = transition
        self.pointer += 1