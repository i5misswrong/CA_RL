import Data, Game
import time
import tensorlayer as tl
import tensorflow as tf
import numpy as np
from tensorlayer.layers import DenseLayer, InputLayer

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

image_size = Data.ROOM_M
size_pow = image_size * image_size

H = 50
batch_size = 10
learning_rate = 1e-3
gamma = 0.99
decay_rate = 0.5
render = False
model_file_name = 'CA'
np.set_printoptions(threshold=np.nan)

env = Game.Game()
observation = env.reset()
prev_x = None
running_reward = None
reward_sum = 0
episode_number = 0

xs, ys, rs = [], [], []
t_states = tf.placeholder(tf.float32, shape=[None, size_pow])
network = InputLayer(t_states, name='input')
network = DenseLayer(network, n_units=H, act=tf.nn.relu, name='hidden')
network = DenseLayer(network, n_units=4, name='output')

probs = network.outputs
sampling_prob = tf.nn.softmax(probs)

t_actions = tf.placeholder(tf.int32, shape=[None])
t_discount_reward = tf.placeholder(tf.float32, shape=[None])

loss = tl.rein.cross_entropy_reward_loss(probs, t_actions, t_discount_reward)

train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess, model_file_name + '.npz', network)
    network.print_params()
    network.print_layers()

    start_time = time.time()
    game_number = 0
    while True:
        env.render()
        cur_x = observation.ravel()

        if prev_x is not None:
            x = cur_x - prev_x
        else:
            x = np.zeros(size_pow)
        x = x.reshape(1, size_pow)
        prev_x = cur_x

        prob = sess.run(sampling_prob, feed_dict={t_states: x})
        action = tl.rein.choice_action_by_probs(prob.flatten(), [0,1,2,3])

        observation, reward, done = env.step(action)
        reward_sum += reward
        xs.append(x)
        ys.append(action)
        rs.append(reward)

        if done:
            episode_number += 1
            game_number = 0

            if episode_number % batch_size == 0:
                print('over batch  updata paramaters')
                epx = np.vstack(xs)
                epy = np.asarray(ys)
                epr = np.asarray(rs)
                disR = tl.rein.discount_episode_rewards(epr, gamma)
                disR -= np.mean(disR)
                disR /= np.std(disR)

                xs, ys, rs = [], [], []
                sess.run(train_op, feed_dict={t_states: epx, t_actions: epy, t_discount_reward: disR})
            if episode_number % (batch_size * 10 ) == 0:
                tl.files.save_npz(network.all_params, name=model_file_name + '.npz')

            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset()
            prev_x = None
        if reward != 0:
            print(
                (
                        'episode %d: game %d took %.5fs, reward: %f' %
                        (episode_number, game_number, time.time() - start_time, reward)
                )
            )
            start_time = time.time()
            game_number += 1