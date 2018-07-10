import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
#GPU参数配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#初始化变量
trX=mnist.train.images
trY=mnist.train.labels
teX=mnist.test.images
teY=mnist.test.labels
#训练参数形状
trX=trX.reshape(-1,28,28,1)
teX=teX.reshape(-1,28,28,1)
##输入输出
X=tf.placeholder(tf.float32,[None,28,28,1])
Y=tf.placeholder(tf.float32,[None,10])

#init weight
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))
#3层卷积层
w=init_weight([3,3,1,32])
w2=init_weight([3,3,32,64])
w3=init_weight([3,3,64,128])
w4=init_weight([128*4*4,625]) #  全链接层
w_o=init_weight([625,10])# 输出层



def model(X,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden):

    # group-001  a11_shape=(?,28,28,32)
    l1a=tf.nn.relu(tf.nn.conv2d(X,w,strides=[1,1,1,1],padding='SAME'))#卷积层
    l1=tf.nn.max_pool(l1a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#池化层
    l1=tf.nn.dropout(l1,p_keep_conv)#设置隐藏的神经元

    # group-002  l2a_shape=(?,28,28,32)
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    # group-003  l3a_shape=(?,28,28,32)
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3=tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    # 全连接层
    l4=tf.nn.relu(tf.matmul(l3,w4))
    l4=tf.nn.dropout(l4,p_keep_hidden)

    # 输出层
    pyx=tf.matmul(l4,w_o)
    return pyx

p_keep_conv=tf.placeholder(tf.float32)
p_keep_hidden=tf.placeholder(tf.float32)

py_x=model(X,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
train_op=tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predict_op=tf.argmax(py_x,1)

batch_size=128
test_size=256

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        training_batch=zip(range(0,len(trX),batch_size),range(batch_size,len(trX)+1,batch_size))
        for start,end in training_batch:
            sess.run(train_op,feed_dict={X:trX[start:end],Y:trY[start:end],p_keep_conv:0.8, p_keep_hidden:0.5})
        test_indices=np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices=test_indices[0:test_size]
        print(i,np.mean(np.argmax(teY[test_indices],axis=1)==sess.run(predict_op,feed_dict={X:teX[test_indices],p_keep_conv:1.0,p_keep_hidden:1.0})))
