from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)#one_hot：是否把标签转为一维向量

print("----------------------------------------------------------")
#55000：55000个样本
#784：每张图片像素为28*28，784维的特征，就是28*28个点展开成1维的结果
#10：10个类别，0-9
#训练集，(55000, 784) (55000, 10)
print(mnist.train.images.shape, mnist.train.labels.shape)
#测试集  (10000, 784) (10000, 10)
print(mnist.test.images.shape, mnist.test.labels.shape)
#验证集  (5000, 784) (5000, 10)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
print("----------------------------------------------------------")

import tensorflow as tf
sess = tf.InteractiveSession()
#shape:[None, 784],None代表不限条数的输入
x = tf.placeholder(tf.float32, [None, 784]) 

W = tf.Variable(tf.zeros([784, 10]))#784是特征的维数，10代表有10类
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
#通常使用cross_entropy作为loss function
#cross_entropy属于计算图中的结点
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#train_step属于计算图中的结点
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#当你调用tf.constant时常量被初始化，它们的值是不可以改变的，而变量当你调用tf.Variable时没有被初始化，
#在TensorFlow程序中要想初始化这些变量，你必须明确调用一个特定的操作，如下：
tf.global_variables_initializer().run()


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#accuracy属于计算图中的结点
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("----------------------------------------------------------")
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))




