# -*- coding: utf-8 -*-
# 添加drop_out操作
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def add_layer(inputs, in_size, out_size, activate_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)
    if activate_function is not None:
        outputs = activate_function(wx_plus_b)
    else:
        outputs = wx_plus_b
    return outputs


def compute_accuracy(x, y):
    global prediction
    predict = sess.run(prediction, feed_dict={xs: x, ys: y, keep_prob: 1})
    # 预测的时候是不需要神经元失活的，因为参数都是已经训练好的，因此keep_prob应该设置为1，设为0.几 则会使得预测的结果很差，训练的模型就没用了
    pre_t_f = tf.equal(tf.math.argmax(predict, 1), tf.math.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pre_t_f, tf.float32))
    return sess.run(accuracy, feed_dict={xs: x, ys: y})


digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30)

keep_prob = tf.placeholder(tf.float32)

xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])


l1 = add_layer(xs, 64, 50, activate_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, activate_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.4})
        if i % 10 == 0:
            train_accuracy = compute_accuracy(x_train, y_train)
            test_accuracy = compute_accuracy(x_test, y_test)
            print train_accuracy, test_accuracy

