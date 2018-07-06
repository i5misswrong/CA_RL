import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
#GPU参数配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

trX=mnist.train.images
print(trX[0].shape)
# (x_train,y_train),(x_test,y_test)=mnist.load_data()
plt.imshow(trX[0].reshape(8,98))# reshape用于改变数组形状 28*28=784 总数保持不变 例如8*98
plt.show()

# X=tf.placeholder(tf.float32,[None,28,28,1])



# (x_train,y_train),(x_test,y_test)=mnist.load_data()
# print(type(trX))



