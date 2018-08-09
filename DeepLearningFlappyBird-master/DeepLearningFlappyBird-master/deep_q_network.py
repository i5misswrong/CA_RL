#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions 动作空间
GAMMA = 0.99 # decay rate of past observations 折扣因子
OBSERVE = 1000. # timesteps to observe before training 训练前观察到的时间步
EXPLORE = 2000000. # frames over which to anneal epsilon  ？？
FINAL_EPSILON = 0.0001 # final value of epsilon ？？
INITIAL_EPSILON = 0.0001 # starting value of epsilon ？？
REPLAY_MEMORY = 50000 # number of previous transitions to remember 之前的记忆数量
BATCH = 32 # size of minibatch  训练批次尺寸
# BATCH=10
FRAME_PER_ACTION = 1

def weight_variable(shape):  #初始化weight
    initial = tf.truncated_normal(shape, stddev = 0.01) #正态分布随机初始化
    return tf.Variable(initial)

def bias_variable(shape): #初始化bias
    initial = tf.constant(0.01, shape = shape) #偏移值为0.01
    return tf.Variable(initial)

def conv2d(x, W, stride): #定义卷积层
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x): #定义2x2池化层
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():#创建神经网络
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])  #第一层卷积
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64]) #第三层卷积
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64]) #第五层卷积
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512]) #第二层池化
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS]) #第四层池化
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])  #屏幕截图80x80x4 4帧

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    '''
    经验池
    D采用了队列的数据结构，是TensorFlow中最基础的数据结构，可以通过dequeue()
    和enqueue([y])
    方法进行取出和压入数据。经验池D用来存储实验过程中的数据，
    后面的训练过程会从中随机取出一定量的batch进行训练。
    '''
    D = deque() #经验池

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    #cv2.resize() 改变图片形状
    #cv2.cvtColor(原始图片，输出图片) cv2.Color_Bgr2gray 转为灰度图
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)

    #cv2.threshold 二值化处理 将灰度图的灰度值设置为0/255 呈现处黑白鲜明的效果 突出轮廓
    #ret为返回的最优阈值  x_t为二值化后的图像
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

    #将输入转为四通道
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()  # 检查点， 用于保存和载入数据
    # sess.run(tf.initialize_all_variables()) #初始化所有变量
    sess.run(tf.global_variables_initializer())  # 初始化所有变量
    checkpoint = tf.train.get_checkpoint_state("saved_networks") #获取之前保存的训练数据
    if checkpoint and checkpoint.model_checkpoint_path: #如果路径存在
        saver.restore(sess, checkpoint.model_checkpoint_path) #载入数据
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights") #抛出异常

    # start training
    epsilon = INITIAL_EPSILON  #初始ε  这个ε是什么。。
    # 根据概率ε来选择一个动作
    t = 0
    while "flappy bird" != "angry bird": #启动游戏
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]#神经网络的输入
        a_t = np.zeros([ACTIONS]) #将动作值全部置零
        action_index = 0 #动作索引？
        if t % FRAME_PER_ACTION == 0: #如果游戏步数不等于=开始了-----此处有问题
            if random.random() <= epsilon:#产生一个随机数，如果小于概率
                print("----------Random Action----------")#使用随机动作
                #动作值空间为[0,1][1,0]等，需要填充2次
                action_index = random.randrange(ACTIONS)#动作索引改为随机动作 0/1
                a_t[random.randrange(ACTIONS)] = 1#动作值空间改为随机动作
            else:#如果随机数大于ε
                action_index = np.argmax(readout_t) #获取readout最大值的索引
                a_t[action_index] = 1 #将该动作值=1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        # 此处 ε的变化使用的都是常量，如何改变
        if epsilon > FINAL_EPSILON and t > OBSERVE: #如果ε大于最终ε 并且时间步长>10W
            # new ε=(0.0001-0.0001)/2000000-ε
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)#开始运行游戏
        # x_t1_colored为图像数据  r_t为reward

        # 转为灰度图
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)

        #二值化处理
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)

        #将图像转为80*80
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)

        #将训练前的s_t的最后一个通道添加到s_t1
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        #将训练数据添加到记忆池
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE: #如果t>2W
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)#使用记忆池中的数据进行最小化皮训练

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
