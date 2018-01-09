#coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#对激活函数参数进行滑动平均，正则化用在loss函数里
mnist = input_data.read_data_sets("/Users/qiuqian/code/MNIST_data/", one_hot=True)
#MNIST数据集相关的常数

INPUT_NODE = 784 #输入层节点数，对于MNIST数据集，相当于图片的像素，转化成一维的数组，28*28
OUTPUT_NODE = 10 #输出节点，这个等于类别的数目，因为需要区分的是0～9这10个数字，因此输出层节点数为10

LAYER1_NODE = 500#隐藏层节点数，这里使用只有一个隐藏层的网络结构作为样例，这个隐藏层有500个节点

BATCH_SIZE = 100#一个训练batch里训练数据的个数

LEARNING_RATE_BASE = 0.8 #基础的学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率

REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

#给定神经网络的输入和所有参数，计算神经网络的前向传播结果，使用了三层全连接神经网络，通过隐藏层实现了多层网络结构，通过ReLU激活函数实现了去线性化
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #当没有提供滑动平均类时，直接使用前向激活函数
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        #计算输出层的前向传播结果，因为在计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数，而且不加入softmax不会影响预测结果
        #因为预测时使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后分类结果的计算没有影响
        return tf.matmul(layer1, weights2) + biases2
    else:#计算变量的滑动平均值，再计算前向传播结果
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(minst):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')#输入
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')#标准结果

    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1)
    )#初始化，正态分布
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    #初始化为0.1
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1)
    )
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    #计算y的前向传播结果，预测值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    #定义存储训练轮数的变量。这个变量不需要进行滑动平均，也设置成神经网络中不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    #给定平均衰减率和训练轮数，创造一个滑动平均的类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )

    #在所有代表神经网络参数的变量上使用滑动平均，trainable_variables返回的是图上集合GraphKeys.TRAINABLE_VARIABLES中的元素，这个集合的元素就是所有没有指定不可优化的参数
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )
    #计算使用滑动平均后的前向传播结果
    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2
    )
    #交叉熵定义损失函数，刻画预测值和真实值之间的差距，当分类问题只有一个正确答案时，可以使用sparse_softmax_cross_entropy_with_logits函数加速交叉熵的计算
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        y, tf.argmax(y_, 1)
    )
    #计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失。只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失为交叉上损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, #基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,    #当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  #过完所有训练数据需要的迭代次数
        LEARNING_RATE_DECAY #学习率衰减速度
    )
    #使用优化算法优化损失函数，损失函数里已经包含了：交叉熵损失 和 L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值，为了一次完成多个操作，使用了group操作
    train_op = tf.group(train_step,variable_averages_op)
    #检验使用了滑动平均模型的神经网络前向传播结果是否正确
    #average_y是一个 batch_size * 10 的而且数组，每一行表示一个样例的前向传播结果，参数1表示选取最大值的操作仅在第一个唯独中进行，
    #也就是说，只在每一行选取最大值的下表，于是得到的结果是一个长度为batch的一维数组，这个一维数组中的值就代表每一个样例对应的数字识别，equal判断两个张量每一维是否相等，相等返回True
    correct_prediction = tf.equal(tf.argmax(average_y,1), tf.argmax(y_, 1))
    #将布尔型转成实数型，然后计算平均值，这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()#初始化所有变量

        validate_feed = {
            x:mnist.validation.images,
            y_:mnist.validation.labels
        }
        #测试数据，
        test_feed = {x:mnist.test.images, y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                #计算训练过程中得到的准确率
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)

                print("after %d training steps,validation accuracy using average model is %g" % (i, validate_acc))
            #不断训练神经网络，并滑动平均参数
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            sess.run(train_op, feed_dict={x:xs,y_:ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        #输出最终的正确率
        print("test accuracy using average model is %g" % (test_acc))
#主程序入口
def main(argv=None):
    train(mnist)
#Tensorflow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()


