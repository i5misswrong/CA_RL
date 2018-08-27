import tensorflow as tf
import gym
import numpy as np
import cv2
import random
from collections import deque

env = gym.make('CartPole-v0')
env.reset()  # 初始化游戏状态
# env.render()  # 执行完该语句后 出现游戏窗口

ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 2000
FINAL_EPSLON = 0.0001
INITIAL_EPSLON = 0.001
REPLAY_MEMORY = 5000
BATCH = 32

FRAME_PER_ACTION = 1


# for i in range(1000):
#     env.render()
#     env.step(env.action_space.sample())

def weight_vaiable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    # 此处参数需要调整
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def creatNetwork():
    # todo set paramater
    W_conv1 = weight_vaiable([80, 80, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_vaiable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_vaiable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_vaiable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_vaiable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    s = tf.placeholder('float', [None, 80, 80, 2])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 2) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


def trainNetwork(s, readout, h_fc1, sess):
    D = deque()

    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_acction = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_acction))
    train_step = tf.train.AdadeltaOptimizer(1e-6).minimize(cost)

    game_state = env.reset()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    observation_, reward, done, info = env.step(do_nothing)

    x_t = cv2.cvtColor(cv2.resize(observation_, (80, 80)), cv2.COLOR_BRG2GRAY)

    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    sess.run(tf.global_variables_initializer())

    epsilon = INITIAL_EPSLON
    t = 0

    while True:
        # TODO set game end state
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----random action-----")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1

        if epsilon > FINAL_EPSLON and t > OBSERVE:
            epsilon -= (INITIAL_EPSLON - FINAL_EPSLON) / EXPLORE

        observation_1, reward_t, done_t, info_t = env.step(do_nothing)

        x_t1 = cv2.cvtColor(cv2.resize(observation_1, (80, 80)), cv2.COLOR_BGR2GRAY)

        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)

        x_t1 = np.reshape(x_t1, (80, 80, 1))

        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)

            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})

            for i in range(0, len(minibatch)):
                done_t = minibatch[i][4]
                if done_t:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch
            })
        s_t=s_t1
        t=t+1

        state=""

def playGame():
    sess=tf.InteractiveSession()
    s,readout,h_fc1=creatNetwork()
    trainNetwork(s,readout,h_fc1,sess)

def main():
    playGame()

if __name__ == '__main__':
    main()