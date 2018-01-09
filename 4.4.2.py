#coding:utf-8

import tensorflow as tf

#获取每一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为'losses'的集合
def get_weight(shape, lambda1):
    #生成一个变量，正太分布初始化，按shape格式
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    #add_to_collection,把新生成的L2正则化损失项加入集合
    tf.add_to_collection(
        'losses',tf.contrib.layers.l2_regularizer(lambda1)(var))

x = tf.placeholder(tf.float32, shape=(None, 2))

y_ = tf.placeholder(tf.float32, shape=(None, 1))

batch_size = 8
#每一层网络中节点个数
layer_dimension = [2, 10, 10, 10, 1]
#神经网络的层数
n_layers = len(layer_dimension)

#维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x
#当前层的节点个数
in_dimension = layer_dimension[0]

for i in range(1,n_layers):#每一层神经网络循环
    #下一层节点个数
    out_dimension = layer_dimension[i]
    #生成当前层节点的权重变量，将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension,out_dimension],0.001)

    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    #使用relu激活函数，激活函数实际上就是前向传播的对应关系
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    #更新当前层
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))#实际值减去预测值，求平方根

tf.add_to_collection('losses',mse_loss)#加入到集合

loss = tf.add_n(tf.get_collection('losses'))#全部累加，就是最终的损失函数

