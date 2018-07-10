import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# GPU参数配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_epochs = 100  # 训练轮数
batch_size = 100  # 随机出去数据大小
display_step = 1  # 显示训练结果的间隔
learning_rate = 0.0001  # 学习效率
drop_prob = 0.5  # 正则化,丢弃比例
fch_nodes = 512  # 全连接隐藏层神经元的个数


# 权重初始化
def weight_init(shape):
    weights = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(weights)


# 偏置初始化
def biases_init(shape):
    biases = tf.random_normal(shape, dtype=tf.float32)
    return tf.Variable(biases)


# 随机选取bitchj
def get_random_batchData(n_samples, batchsize):
    start_index = np.random.randint(0, n_samples - batch_size)
    return (start_index, start_index + batch_size)


# 全链接层 权重初始化
def xavier_init(layer1, layer2, constant=1):
    Min = -constant * np.sqrt(6.0 / (layer1 + layer2))
    Max = constant * np.sqrt(6.0 / (layer1 + layer2))
    return tf.Variable(tf.random_uniform((layer1, layer2), minval=Min, maxval=Max, dtype=tf.float32))


# 卷积
def conv2d(x, w):
    # x为图形像素 w为卷积核
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化
def max_pool_2x2(x):
    # x是卷积后 经过激活函数后的图像 ksize的池化滑动张量 ksize 的维度[batch, height, width, channels],跟 x 张量相同
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 创建占位符  x是图像  y是标签！！！
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 将一维图像转化成二维
x_img = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积+池化
w_conv1 = weight_init([5, 5, 1, 16])
b_conv1 = biases_init([16])
h_conv1 = tf.nn.relu(conv2d(x_img, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积+池化
w_conv2 = weight_init([5, 5, 16, 32])
b_conv2 = biases_init([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv1)

# 全连接层
h_fpool2 = tf.reshape(h_pool2, [-1, 7 * 7 * 32])

w_fc1 = xavier_init(7 * 7 * 32, fch_nodes)
b_fc1 = biases_init([fch_nodes])
h_fc1 = tf.nn.relu(tf.matmul(h_fpool2, w_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=drop_prob)

w_fc2 = xavier_init(fch_nodes, 10)
b_fc2 = biases_init([10])

y_ = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)
y_out = tf.nn.softmax(y_)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_out), reduction_indices=[1]))

#
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_out, 1))
accutacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

n_samples = int(mnist.train.num_examples)
total_batches = int(n_samples / batch_size)

with tf.Session(config=config) as sess:
    sess.run(init)
    # Cost=[]
    # Accuracy=[]
    # for i in range(train_epochs):
    #     for j in range(100):
    #         start_index,end_index=get_random_batchData(n_samples,batch_size)
    #         batch_x=mnist.train.images[start_index:end_index]
    #         batch_y=mnist.train.labels[start_index:end_index]
    #         sess.run([optimizer,cross_entropy,accutacy],feed_dict={x:batch_x,y:batch_y})
    #         Cost.append()
    #
    #

    iput_imgae=mnist.train.images[0]
    iput_imgae_2=mpimg.imread('111.png')
    iput_imgae_2.shape
    plt.imshow(iput_imgae_2)
    plt.show()
    # conv1_16=sess.run(h_conv1,feed_dict={x:iput_imgae})
    # conv1_traonspose=sess.run(tf.transpose(conv1_16,[3,0,1,2]))
    # fig,ax=plt.subplot(rows=1,cols=16,figzise=(16,1))
    # for i in range(16):
    #     ax[i].imshow(conv1_traonspose[i][0])
    # print(len(iput_imgae))
    # img_shape=iput_imgae.reshape(28,28)
    # # for i in iput_imgae[3][0]:
    # #     print(i)
    # # plt.imshow(conv1_traonspose[3][0])
    # plt.imshow(img_shape)
    # plt.show()