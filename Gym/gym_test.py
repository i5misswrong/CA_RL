"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
import DeepQNet

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space) #2
print(env.observation_space) #4
print(env.observation_space.high)
print(env.observation_space.low)

print(env.action_space.n,'---')

RL = DeepQNet.DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0


for i_episode in range(100):

    observation = env.reset() #初始化游戏状态
    ep_r = 0
    while True:
        env.render() #执行完该语句后 出现游戏窗口

        #获取动作
        action = RL.choose_action(observation)
        #获取训练数据
        # 环境 期望值 动作 信息（可选）
        observation_, reward, done, info = env.step(action)

        #修改期望值
        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5

        #期望为一正一负
        reward = r1 + r2

        #存入记忆池
        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
