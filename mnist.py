# 下载mnist集合
from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)
import tensorflow as tf
# 创建会话环境
sess = tf.InteractiveSession()
# 输入数据x的占位符
x = tf.placeholder(tf.float32, [None, 784])
# 给权重创建变量对象
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 输出y的计算
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 初始化y的占位符
y_ = tf.placeholder(tf.float32, [None, 10])
# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# tensonflow 全局参数初始化器，运行它
tf.global_variables_initializer().run()
# 迭代得进行训练操作
for i in range(5000):
    batvh_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batvh_xs, y_: batch_ys})
# 验证准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 统计全部样本预测的准确率 accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 打印准确率结果
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
