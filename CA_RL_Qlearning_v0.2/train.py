import random, sys
import Game, Bulid_network
import numpy as np
import tensorlayer as tl
import tensorflow as tf
from  collections import deque


learning_rate = 1e-2
initial_spsilon = 0.99
explore_number = 1000
observe_number = 1000
replay_memory = 10000
batch_size = 100


def play():
    env = Game.Game()
    observation = env.reset()
    pre_x = None
    running_reward = 0
    reward_sum = 0
    t_state = observation
    # t_state = tf.placeholder(tf.float32, shape=[None, self.mat_pow])
    predict_action = Bulid_network.build().build_net()

if __name__ == '__main__':
    pass
