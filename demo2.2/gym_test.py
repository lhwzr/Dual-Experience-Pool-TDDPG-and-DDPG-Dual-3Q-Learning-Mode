# python 的学习测试
# 基础学习
import gym
import numpy as np
from DDPG import DDPG
model = DDPG(a_dim=1, s_dim=3, a_bound=[-2, 2], batch_size=128, tau=0.01, gamma=0.9, memory_capacity=10000)

env = gym.make('Pendulum-v1')
RENDER = False  # 是否渲染环境
env = env.unwrapped  # 取消限制
# 循环次数200*50 = 10000 最大记忆库保存数
max_ep_step = 200

var = 3
for ep in range(200):
    s = env.reset()  # 环境初始化 设置随机种子
    # print(s)
    ep_reward = 0
    for step in range(max_ep_step):
        if RENDER:
            env.render()
        # 增加探索时的噪音
        a = model.act(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # 为行动选择添加随机性进行探索，action超过[-2,2]时做截断处理
        # print(a)
        s_, r, done, info = env.step(a)

        # 将当前的状态,行为,回报,下一个状态存储到记忆库中
        model.store_transition(s, a, r/10, s_)

        # 达到记忆库容量的最大值
        if model.pointer > 10000:
            var *= .9995  # 衰减动作随机性
            model.learn()  # 开始学习
            model.param_replace()  # 参数更新

        s = s_
        ep_reward += r
        if step == max_ep_step -1:
            print('Episode:', ep, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:  # 达到回合最大值且回合回报值大于-300,渲染环境
                RENDER = True
            break

env.close()  # 关闭渲染窗口