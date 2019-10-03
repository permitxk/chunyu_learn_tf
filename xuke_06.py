# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 先定义一波已经决定的网络格式的参数
hidden_cells = 128
batch_size = 128
num_steps = 28
num_inputs = 28
num_classes = 10
learning_rate = 0.1
training_iter = 100000

x = tf.placeholder(tf.float32, [None, num_steps*num_inputs])
y = tf.placeholder(tf.float32, [None, num_classes])

weights = {
    'in': tf.Variable(tf.random_normal([num_inputs, hidden_cells])),
    'out': tf.Variable(tf.random_normal([hidden_cells, num_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[hidden_cells, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[num_classes, ]))
}

# 定义一个函数，使得输入数据，进行相应的运算，得到rnn网络的输出结果，因此传入的参数应该是数据和参与运算的变量们


def RNN(X, weights, biases):
    """
    :param X: 输入数据，shape[batch_size,784],要和weights做矩阵乘法，需要对维数做调整，调成合适的二维数组
    :param weights:变量
    :param biases:变量
    :return: 输出值 10维的
    """
    X = tf.reshape(X, [-1, num_inputs])  # 维度：[batch_size*28, 28]
    X_in = tf.matmul(X, weights['in']) + biases['in']  # 维度：[batch_size*28, 128] broadcasting
    X_in = tf.reshape(X_in, [-1, num_steps, hidden_cells])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_cells, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results


prediction = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
correct_pre = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iter:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #batch_xs = batch_xs.reshape([batch_size, num_steps, num_inputs])
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        step += 1


