#coding:utf-8
#滑动平均
import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)
#模拟神经网络中迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)
#定义一个滑动平均的类，初始化时给定了衰减率和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
#定义一个更新变量滑动平均的操作，这里需要有一个列表，每次执行这个操作，列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
#输出原v1和滑动平均之后的v1
    print (sess.run([v1, ema.average(v1)]))
#设置v1的值后，再次放入列表
    sess.run(tf.assign(v1, 5))

    sess.run(maintain_averages_op)
#获取滑动平均之后的值，衰减率为min{decay,(1 + step)/(10 + step)}
    print(sess.run([v1, ema.average(v1)]))
#shadow_variable = decay*shadow_variable + (1-decay)*variable，影子变量是每次都更新的值，variable是放入列表中的值
    sess.run(tf.assign(step, 10000))

    sess.run(tf.assign(v1, 10))

    sess.run(maintain_averages_op)

    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_averages_op)

    print(sess.run([v1, ema.average(v1)]))