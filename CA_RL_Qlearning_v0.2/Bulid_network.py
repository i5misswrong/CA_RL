import tensorflow as tf
import tensorlayer as tl
import numpy as np

import Game

class build():
    def __init__(self):
        self.mat_pow = Game.ROOM_M ** 2
        self.n_features = 2 # 特征 游戏运行 / 游戏结束
        self.n_actions = 0
        self.learning_rate = 1e-2
        self.e_greedy = 0.9
        self.gamma = 0.99
        self.replace_target_iter = 300
        self.memory_size = 500
        self.batch_size = 32
        self.memory_counter = 0
        self.e_greedy_increment = None

        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.sess = tf.Session()
        self.learn_step_counter = 0
        self.cost_his = []

        if self.e_greedy_increment is not None:
            self.epsilon = 0
        else:
            self.epsilon = self.e_greedy

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]



    def build_net_tl(self, t_state):
        # t_state = tf.placeholder(tf.float32, shape=[None, self.mat_pow])
        network = tl.layers.InputLayer(t_state, name='input')
        network = tl.layers.DenseLayer(network, n_units=100, act=tf.nn.relu, name='hidden')
        network = tl.layers.DenseLayer(network, n_units=4, act=tf.nn.relu, name='output')

        return  network

    def build_net_tf_dense(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')

        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        '''
        inputs：张量输入。
        units：整数或长整数，输出空间的维数。
        activation：激活功能（可调用）。将其设置为“无”以保持线性激活。
        use_bias：Boolean，该层是否使用偏差。
        kernel_initializer：权重矩阵的初始化函数。如果None（默认），使用默认初始化程序初始化权重tf.get_variable。
        bias_initializer：偏置的初始化函数。
        kernel_regularizer：权重矩阵的正则化函数。
        bias_regularizer：正规函数的偏差。
        activity_regularizer：输出的正则化函数。
        kernel_constraint：由a更新后应用于内核的可选投影函数Optimizer（例如，用于实现层权重的范数约束或值约束）。该函数必须将未投影的变量作为输入，并且必须返回投影变量（必须具有相同的形状）。在进行异步分布式培训时，使用约束是不安全的。
        bias_constraint：由a更新后应用于偏置的可选投影函数Optimizer。
        trainable：Boolean，如果True还将变量添加到图集合中
        GraphKeys.TRAINABLE_VARIABLES（请参阅参考资料tf.Variable）。
        name：String，图层的名称。
        reuse：Boolean，是否以同一名称重用前一层的权重。
        '''
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_features, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='q')
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_features, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1) #强制转换类型
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices) #incices为整数 所以需要上一步强制转化类型 具体没理解

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_x_')
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_ERROR'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, obversation):
        obversation = obversation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: obversation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('target_params_replaced')
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, self=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:,:self.n_features],
                self.a: batch_memory[: self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_ :batch_memory[:,-self.n_features:],
            }
        )

        self.cost_his.append(cost)
        #todo 贪婪值需要更改
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon else self.epsilon
        self.learn_step_counter += 1