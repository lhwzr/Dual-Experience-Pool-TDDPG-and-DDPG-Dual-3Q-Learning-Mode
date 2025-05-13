# python 的学习测试
# 基础学习
import numpy as np
import tensorflow as tf

class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        self.gamma = 0.99
        self.tau = 0.01
        self.memory_size = 10000
        self.batch_size = 32
        self.memory = np.zeros((self.memory_size, state_dim * 2 + action_dim + 12), dtype=np.float32)
        self.pointer = 0
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)

    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(30, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.action_dim, activation='tanh')
        ])
        return model

    def build_critic(self):
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        action_input = tf.keras.layers.Input(shape=(self.action_dim,))
        x = tf.keras.layers.Concatenate()([state_input, action_input])
        x = tf.keras.layers.Dense(30, activation='relu')(x)
        x = tf.keras.layers.Dense(30, activation='relu')(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=x)
        return model

    def choose_action(self, state):
        return self.actor.predict(state)

    def learn(self):
        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        ba = bt[:, self.state_dim:self.state_dim + self.action_dim]
        br = bt[:, -self.state_dim - 1: -self.state_dim]
        bns = bt[:, -self.state_dim:]

        with tf.GradientTape() as tape:
            action = self.actor(bs)
            q = self.critic([bs, action])
            actor_loss = -tf.reduce_mean(q)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            next_action = self.target_actor(bns)
            next_q = self.target_critic([bns, next_action])
            target_q = br + self.gamma * next_q
            q = self.critic([bs, ba])
            critic_loss = tf.reduce_mean(tf.square(q - target_q))

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        self.update_target_networks()

    def update_target_networks(self):
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]

        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]

        self.target_actor.set_weights(target_actor_weights)
        self.target_critic.set_weights(target_critic_weights)

    def store_transition(self, state, action, reward, next_state):
        transition = np.concatenate((state, action, reward, next_state), axis=None)
        index = self.pointer % self.memory_size
        self.memory[index, :] = transition.flatten()
        self.pointer += 1

    def train(self, data):
        state_dim = 60  # 状态维度
        action_dim = 12  # 动作维度
        action_bound = 1  # 动作的范围

        ddpg = DDPG(state_dim, action_dim, action_bound)

        # 平移取出数据
        data_shifted = []
        for i in range(len(data_train) - 5):
            data_shifted.append(data_train[i:i + 5, :].flatten())
        data_shifted = np.array(data_shifted)
        # 数据预处理
        data_shifted = (data_shifted - np.mean(data_shifted, axis=0)) / np.std(data_shifted, axis=0)
        # 第一步打印平移数据
        print(data_shifted, '11111',data_shifted.shape)
        # 定义RL状态和动作
        # state = np.zeros((1, 60))  # 初始状态
        # action = np.zeros((1, 12))  # 初始动作
        #定义总奖励值
        reward_c = []
        for i in range(len(data_shifted) - 1):
            if i == len(data_shifted) - 1:
                # 计算奖励
                reward = -np.abs(action - np.reshape(data_shifted[i + 1, 47:59],(1,12)))
                reward_c.append(reward)
                print(reward_c,'33333')
            else:
                # state[:, :-12] = state[:, 12:]
                state = np.reshape(data_shifted[i, :], (1, 60))
                print(state,state.shape,'555555')
                # print('state_shape:',state,state.shape)
                # 预测下一时刻状态作为动作
                action = ddpg.choose_action(state)
                print(action,action.shape,'666666666')
                next_state = np.reshape(data_shifted[i+1, :], (1, 60))
                # next_state[:, -12:] = action

                # 计算奖励  取出真实值的12个数与其预测值进行误差计算，获得奖励值
                reward = -np.abs(action - np.reshape(data_shifted[i + 1, 47:59],(1,12)))
                reward_c.append(reward)
                print(reward_c,'222',len(reward_c))
                # 存储经验并训练
                ddpg.store_transition(state.flatten(), action.flatten(), reward.flatten(), next_state.flatten())
                if ddpg.pointer > ddpg.memory_size:
                    ddpg.learn()

                state = next_state

        # 预测结果
        predictions = []
        # 测试数据平移取出数据
        data_shifted_test = []
        for i in range(len(data_test) - 5):
            data_shifted_test.append(data_test[i:i + 5, :].flatten())
        data_shifted_test = np.array(data_shifted_test)
        # 数据预处理
        data_shifted_test = (data_shifted_test - np.mean(data_shifted_test, axis=0)) / np.std(data_shifted_test, axis=0)
        print(data_shifted_test, '4444', data_shifted_test.shape)
        for i in range(len(data_test)-1):
            state = data_shifted_test[i, :]
            action = ddpg.choose_action(state)
            predictions.append(action.flatten())

        return predictions

# 示例数据
data = np.random.random((1000, 12))
train = 700
test = 300
data_train = data[0:train,:]
data_test = data[train:-1,:]

# 训练和预测
ddpg = DDPG(60, 12, 1)
predictions = ddpg.train(data)
print(predictions,predictions.shape)