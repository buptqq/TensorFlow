#-*- coding:utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_varibale(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights = tf.get_weight_varibale(
            [INPUT_NODE, LAYER1_NODE],regularizer
        )
        biases = tf.get_weight_varibale(
             "biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
    with tf.variable_scope('layer2'):
        weights = tf.get_weight_varibale(
            [LAYER1_NODE, OUTPUT_NODE],regularizer
        )
        biases = tf.get_weight_varibale(
            "biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.nn.relu(tf.matmul(layer1,weights) + biases)