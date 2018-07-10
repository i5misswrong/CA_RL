import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
# mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# define 回归模型

x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.matmul(x,W)+b

#定义损失函数和优化器
y_=tf.placeholder(tf.float32,[None,10])

#计算y与y_之间的误差
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
#采用SGD作为优化器
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#交互式会话
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()


for _ in range(1000):
    #随机抓取100个数据

    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#模型评估
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
