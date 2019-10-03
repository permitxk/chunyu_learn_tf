# -*- coding:utf-8 -*-
from xuke_03 import add_layer
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()


def compute_accuracy(x, y):
    y_pre = sess.run(prediction, feed_dict={xs: x})
    # tf.math.argmax() 输入一个多维数组，按照某个维度，找出对应最大值的索引
    correct_prediction = tf.equal(tf.math.argmax(y_pre, 1), tf.math.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: x})
    return result


#data = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        xs_batch, ys_batch = mnist.train.next_batch(100)
        # mnist.train.next_batch(100)的类型是tuple，[0]:100*784 [1]:100*10
        sess.run(train_step, feed_dict={xs: xs_batch, ys: ys_batch})
        if i % 100 == 0:
            #data.append(compute_accuracy(mnist.train.images, mnist.train.labels))
            print compute_accuracy(mnist.train.images, mnist.train.labels)

#print data
