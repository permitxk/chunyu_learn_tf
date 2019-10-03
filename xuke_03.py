# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 五、激活函数

# 六、加一层全连接


def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    :param inputs: 输入
    :param in_size: 输入维度，而不是输入数组的规模大小，要区分
    :param out_size: 输出维数
    :param activation_function:激活函数
    :return: 下一层的输出
    """
    weights = tf.Variable(tf.random_normal([in_size, out_size]))  #均值为0 方差为1的全连接随机数
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is not None:
        outputs = activation_function(wx_plus_b)
    else:
        outputs = wx_plus_b
    return outputs

# 七、 构建一个神经网络
# 假数据：x_data, y_data


x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# xs 取placeholder 可以方便的替换训练集
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 仅仅加了一个隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(l1, 10, 1, activation_function=None)
# 损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.1)
# 最小化损失
train = optimizer.minimize(loss)
# 初始化变量，即使变量是出现在定义的函数里面，也需要初始化
init = tf.global_variables_initializer()
# 创建会话
sess = tf.Session()
# 执行初始化
sess.run(init)
# 开始训练
for i in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        print sess.run(loss, feed_dict={xs: x_data, ys: y_data})
# 虽然loss这个张量里面，看似仅有ys一个是placeholder，但是其中prediction-l1-xs，还是有xs这个placeholder的

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
#plt.ion()
plt.show()

for i in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=4)
        plt.ion()
        plt.show()
        plt.pause(0.1)


