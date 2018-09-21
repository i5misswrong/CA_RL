import time
import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import DenseLayer, InputLayer

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

image_size = 80
D = image_size * image_size
H = 200
batch_size = 10
learning_rate = 1e-4
gammma = 0.99
decay_rate = 0.99
render = False
model_file_name = 'model_pong'
np.set_printoptions(threshold=np.nan)


def prepro(I):
    '''
    将obversetion转为一维数据
    :param I: obversition
    :return:
    '''
    #该方法作用貌似是获取球的位置
    I = I[35:195] # 取出 I[35:195]
    I = I[::2, ::2, 0] #
    I[I == 144] = 0 # I 中 144=0
    I[I == 109] = 0 # I 中 109 = 0
    I[I != 0] = 1   # I 填充0
    # astype 强制转化数据类型 转为float
    #ravel 转化为一维数据
    return I.astype(np.float).ravel()


env = gym.make("Pong-v0")
obseration = env.reset()
prev_x = None
running_reward = None
reward_sum = 0
episode_number = 0

xs, ys, rs = [], [], []
t_states = tf.placeholder(tf.float32, shape=[None, D]) #输入s
network = InputLayer(t_states, name='input')#输入层
network = DenseLayer(network, n_units=H, act=tf.nn.relu, name='hidden') #全连接层1
network = DenseLayer(network, n_units=3, name='output')#全连接层2
'''
prev_layer (Layer) -- Previous layer. 上一层
n_units (int) -- The number of units of this layer. #这一层的单位 貌似是输入维度
act (activation function) -- The activation function of this layer. 激活函数
W_init (initializer) -- The initializer for the weight matrix.
b_init (initializer or None) -- The initializer for the bias vector. If None, skip biases.
W_init_args (dictionary) -- The arguments for the weight matrix initializer.
b_init_args (dictionary) -- The arguments for the bias vector initializer.
name (a str) -- A unique layer name.
'''
probs = network.outputs #获取输出层

sampling_prob = tf.nn.softmax(probs) #激活

t_actions = tf.placeholder(tf.int32, shape=[None]) #输入动作
t_discount_reward = tf.placeholder(tf.float32, shape=[None])#reward

loss = tl.rein.cross_entropy_reward_loss(probs, t_actions, t_discount_reward) #计算损失函数
'''
logits (tensor) -- The network outputs without softmax. This function implements softmax inside.
actions (tensor or placeholder) -- The agent actions.
rewards (tensor or placeholder) -- The rewards.
'''
train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss) #训练

with tf.Session() as sess:
    tl.layers.initialize_global_variables(sess) #初始化参数
    tl.files.load_and_assign_npz(sess, model_file_name + '.npz', network) #保存训练过程
    network.print_params()
    network.print_layers()

    start_time = time.time()
    game_number = 0
    while True:
        # if render:
        #     env.render()
        env.render()
        cur_x = prepro(obseration) #将obseration一维化处理

        # 如果prev_x !=None  x=cur_x-prev_x
        # else               x=np.zeros(D)  D = image_size * image_size = 6400
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        x = x.reshape(1, D) #将x转为一维
        prev_x = cur_x  #这里貌似 cur是现实  prev是预测

        prob = sess.run(sampling_prob, feed_dict={t_states: x}) #且softmax激活
        action = tl.rein.choice_action_by_probs(prob.flatten(), [1, 2, 3]) # prob-概率分布  flatten -展开 一维化处理
        '''
        根据概率和动作列表来选动作
        probs (list of float.) -- The probability distribution of all actions.
        action_list (None or a list of int or others) -- A list of action in integer, 
        string or others. If None, returns an integer range between 0 and len(probs)-1.
        '''

        obseration, reward, done, _ = env.step(action) #获取 环境 期望 终止符
        reward_sum += reward #期望总和
        xs.append(x) # x是6400 一维列表 现实值-预测值
        ys.append(action - 1) # t是动作列表
        rs.append(reward) # 期望

        if done: # 如果程序终止
            episode_number += 1 # 迭代计数器+1
            game_number = 0 # 游戏内计时器重置

            if episode_number % batch_size == 0: # 去除batch_size 个数据 from D
                print('batch over   updating paramaters')
                epx = np.vstack(xs) #将xs垂直堆叠
                epy = np.asarray(ys) #将ys赋值给epy 无拷贝
                epr = np.asarray(rs)
                disR = tl.rein.discount_episode_rewards(epr, gammma)
                '''
                获取1D浮动奖励阵列并计算一集的折扣奖励。当输入非零值时，请考虑作为剧集的结尾a。
                reward（列表） - 奖励列表
                gamma（float） - 折扣因子
                mode（int） - 计算折扣奖励的模式。
                            如果mode == 0，则在设置非零奖励（乒乓球比赛）时重置折扣过程。
                            如果mode == 1，则不会重置折扣过程。
                '''
                disR -= np.mean(disR) #np.mean(disR) 求disR的平均值
                disR /= np.std(disR) # np.std(disR) 计算矩阵的标准差

                xs, ys, rs = [], [], [] #重置

                # 运行训练 输入参数： x a r
                sess.run(train_op, feed_dict={t_states: epx, t_actions: epy, t_discount_reward: disR})

            # 到一定时间步后写入训练过程
            if episode_number % (batch_size * 100) == 0:
                tl.files.save_npz(network.all_params, name=model_file_name + '.npz')

            # if running_reward ！= None   running_reward=reward_sum
            # else                       running_reward=running_reward * 0.99 + reward_sum * 0.01
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            obseration = env.reset()
            prev_x = None
        if reward != 0:
            print(
                (
                        'episode %d: game %d took %.5fs, reward: %f' %
                        (episode_number, game_number, time.time() - start_time, reward)
                ), ('' if reward == -1 else ' !!!!!!!!')
            )
            start_time = time.time()
            game_number += 1
