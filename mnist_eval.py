#-*- coding:utf-8 -*-

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input'
        )
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_input'
        )
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #cast函数可以将bool值变为实数型，再计算平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #滑动平均类
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        #滑动平均的该函数可以直接提供所有的变量重命名字典
        variable_to_restore = variable_averages.variables_to_restore()
        #载入模型即可，现在所取的变量都是滑动平均之后的值
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                #get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                #加载模型
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #model.ckpt-1000，获取模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    print("after %s training steps, validation accuracy = %g." % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            #间隔10s调用一次计算正确率的过程检测正确率的变化
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("/Users/qiuqian/code/MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()