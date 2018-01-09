#coding:utf-8
#TensorFlow 3.4.5 训练神经网络解决二分类问题

import tensorflow as tf
import sys

from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))#第一次参数
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))#

x = tf.placeholder(tf.float32, shape=(None,2), name = 'x-input')#
y_ = tf.placeholder(tf.float32, shape=(None,1), name = 'y-input')#y_真实结果

#线性关系
#a = tf.matmul(x,w1)
#y = tf.matmul(a,w2)#预测值，定义神经网络的结构
#relu关系
biases1 = 1
biases2 = 1
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)

cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))#计算预测值和真实值之间的交叉熵
)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)#反向优化算法，不断优化参数

rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size, 2)

Y = [[int(x1 + x2) < 1] for (x1,x2) in X]
#创建数据集
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()#张量初始化，图+张量+会话
    sess.run(init_op)

    print (sess.run(w1))
    print (sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size,dataset_size)

        sess.run(train_step,
                 feed_dict={x: X[start:end],y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X,y_: Y}
            )
            print("After %d training steps, cross entropy on all date is %g" % (i,total_cross_entropy))
    print (sess.run(w1))
    print (sess.run(w2))

