# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

# 一、 Structure
x_data = np.random.rand(100).astype(np.float)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases

# 定义损失
loss = tf.reduce_mean(tf.square(y - y_data))

# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 训练
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(Weights), sess.run(biases)

# 二、 Session

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print "result: ", result
sess.close()

with tf.Session() as sess:
    result2 = sess.run(product)
    print "result: ", result2




