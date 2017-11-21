import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv0 = weight_variable([5, 5, 1, 32])
b_conv0 = bias_variable([32])
h_conv0 = tf.nn.relu(conv2d(x_image, W_conv0) + b_conv0)
h_pool0 = max_pool_2x2(h_conv0)

W_conv1 = weight_variable([5, 5, 32, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_fc0 = weight_variable([7*7*64, 1024])
b_fc0 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool1, [-1, 7*7*64])
h_fc0 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc0) + b_fc0)

W_fc1 = weight_variable([1024,10])
b_fc1 = bias_variable([10])

y_conv = tf.matmul(h_fc0, W_fc1) + b_fc1

"""
Here we use tf.Session instead of tf.InteractiveSession.

This better separates the process of creating the graph and the process of
evaluating the graph. It generally makes for cleaner code.

The tf.Session is created within a 'with' block so that it is automatically
destoryed once the block is exited

"""
cross_entropy = tf.reduce_mean( \
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))