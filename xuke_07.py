# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


def calculate_accuracy(test_x, test_y):
    predict = sess.run(prediction, feed_dict={x: test_x, y: test_y, keep_prob: 1})
    res = tf.equal(tf.arg_max(predict, 1), tf.arg_max(test_y, 1))
    accuracy = tf.reduce_mean(tf.cast(res, dtype=tf.float32))
    result = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1})
    return result


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def conv2d(xx, w):
    return tf.nn.conv2d(xx, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_2x2(h_con1):
    return tf.nn.max_pool(h_con1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])
# 灰度图像，通道数为1

w_con1 = weight_variable([5, 5, 1, 32])
# 第一层的filter 通道数为1， filter组数为32，每组1个卷积核，形状为5*5
b_con1 = weight_variable([32])


h_con1 = tf.nn.relu(conv2d(x_image, w_con1) + b_con1)
# h_con1 的维度为：None * 28 * 28 * 32
h_pool1 = max_pooling_2x2(h_con1)
# 池化之后， h_pool1 的维度为: None * 14 * 14 * 32

# 第二层卷积
w_con2 = weight_variable([5, 5, 32, 64])
# 第二层的filter 通道数为上一层经过池化之后的通道数32，形状为5 * 5，这一层的filter有64组
b_con2 = weight_variable([64])

h_con2 = tf.nn.relu(conv2d(h_pool1, w_con2) + b_con2)
# 第二层卷积之后，维度为： None * 14 * 14 * 64
h_pool2 = max_pooling_2x2(h_con2)
# 第二层池化之后，7 * 7 * 64

# 将池化之后的res 展开，flatter, 接一层普通的神经网络，加上dropout操作
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
w_fl_1 = weight_variable([7*7*64, 1024])
b_fl_1 = weight_variable([1024, ])

h_fl_1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fl_1) + b_fl_1)
h_fl_1_drop = tf.nn.dropout(h_fl_1, keep_prob)

# 添加最后一层softmax,只需要把激活函数从relu 改为softmax即可
w_fl_2 = weight_variable([1024, 10])
b_fl_2 = weight_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fl_1_drop, w_fl_2) + b_fl_2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
train = tf.train.AdadeltaOptimizer(0.1).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        if i % 20 == 0:
            print i, calculate_accuracy(mnist.test.images, mnist.test.labels)