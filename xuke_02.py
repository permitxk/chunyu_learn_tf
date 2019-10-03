# -*- coding:utf-8 -*-

import tensorflow as tf

# 三、 Variable

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))

# 四、 Placeholder

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
# 先定义好运算，然后在用会话sess = tf.Session(),sess.run()去执行
with tf.Session() as sess:
    print sess.run(output, feed_dict={input1: [7., 2.], input2: [2., 7.]})

# 比较下 tf.Variable() & tf.placeholder

input1 = tf.Variable(32.)
input2 = tf.Variable(2.)
output = tf.multiply(input1, input2)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print sess.run(output)




