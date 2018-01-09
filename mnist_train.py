#-*- coding:utf-8 -*-

import os

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100 #每次选取的训练集的数据条数
LEARNING_RATE_BASE = 0.8 #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率
REGULARAZTION_RATE = 0.0001 #正则化的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均的衰减率

MODEL_SAVE_PATH = '/Users/qiuqian/code/MNIST_model/'
MODEL_NAME = "model.ckpt"

def train(mnist):
    #定义输入的数据集和监督学习的结果，正确答案
    x = tf.placeholder(
        tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input'
    )
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input'
    )
    #定义正则化的函数，之后乘权重即可
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #求前向传播的值，并且将正则化损失加入到losses集合
    y = mnist_inference.inference(x, regularizer)
    #定义训练轮数
    global_step = tf.Variable(0, trainable=False)
    #定义滑动平均的类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    #给定一个列表，每次这个列表里的值都会被更新滑动平均值
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #交叉熵，损失函数，经过softmax层之后，变为概率分布
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_ ,1))
    #求出损失函数平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #总的损失=损失函数+正则化损失项
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,#基础学习率
        global_step,#目前轮数
        mnist.train.num_examples / BATCH_SIZE,#总训练次数
        LEARNING_RATE_DECAY #学习率衰减速度
    )
    #定义梯度下降的优化函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #同时优化loss和滑动平均值
    train_op = tf.group(train_step, variable_averages_op)
    #初始化tf的持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #初始化所有变量
        tf.initialize_all_variables().run()
        #训练过程中不再测试模型在验证数据上的表现，验证和测试将由一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step],
                feed_dict={x: xs,y_: ys}
            )
            #保存当前的模型，这里给出了gloabl_step参数，可以让每个被保存模型的文件名末尾加上训练的轮数，如"model.ckpt-1000"表示1000轮后得到的模型
            if i % 1000 == 0:
                print("after %d training steps, loss on training batch is %g." % (step, loss_value))

                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step
                )
def main(argv=None):
    mnist = input_data.read_data_sets("/Users/qiuqian/code/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()