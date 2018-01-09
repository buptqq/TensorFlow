# -*- coding:utf-8 -*-
#神经网络的前向计算过程
import tensorflow as tf

INPUT_NODE = 784#输入节点的元素个数，28*28，变为一维数组
OUTPUT_NODE = 10#输出为10个元素
LAYER1_NONE = 500#隐藏层节点个数

#通过 get_variable函数来获取变量，在训练神经网络时会创建变量；在测试时会通过保存的模型加载这些变量的值。
#更为方便的是，因为可以在变量加载时酱滑动平均变量重命名，所以可以直接用同样的名字在训练时使用变量自身，而在测试时使用滑动平均后的值。
#这个函数中会讲损失函数的正则化部分加入losses集合
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer = tf.truncated_normal_initializer(stddev=0.1)
    )

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))#讲正则化的损失加入损失集合
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):#从输入层到隐藏层
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NONE],regularizer
        )
        #这里通过get_variable或者Variable没有本质区别，因为在训练或是测试中没有在同一个程序中多次调用这个函数，如果在一个程序中多次调用，在第一次调用后需要讲reuse参数设置为True
        biases = tf.get_variable(
            "biases",[LAYER1_NONE],
            initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)#激活函数去线性化

    with tf.variable_scope('layer2'):#从隐藏层到输出层
        weights = get_weight_variable(
            [LAYER1_NONE, OUTPUT_NODE], regularizer
        )
        biases = tf.get_variable(
            "biases",[OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
