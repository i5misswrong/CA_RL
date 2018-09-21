import numpy as np
import tensorflow as tf


class creatNN():


    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate,
                 reward_decay,
                 e_greedy,
                 replace_target_iter,
                 memory_size,
                 batch_size,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon = 0
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

    def build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')

        with tf.variable_scope('eval_net'):
            c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0.0, 0.3)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_name)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_name)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_name)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_name)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_name = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_name)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_name)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_name)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_name)
                self.q_next = tf.matmul(l1, w2) + b2


    def store_trainsitin(self,s, a, r, s_):
        if not hasattr('memory_counter'):
            self.memory_counter = 0
        trainsition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = trainsition
        self.memory_counter += 1


    def choose_action(self,observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_iter)
