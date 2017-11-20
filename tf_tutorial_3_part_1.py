from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
Tensorflow relies on a highly efficient C++ backend to do its computation. The
connection to this backend is called a 'Session'. The common usage for TensorFlow
programs is to first create a graph and then launch it in a 'Session'.

Here instead we use the convenient 'InteractiveSession' class, which makes
Tensorflow more flexible about how you structure your code. It allows you to
interleave operations which build a computation graph with ones that run the graph.

If you are not using an InteractiveSession, then you should build the entire
computation graph before starting a session and launching the graph.
"""

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

"""
If sess = tf.Session() then only

1) sess.run(tf.global_variables_initializer())

works to initialise variables.

However, if sess = tf.InteractiveSEssion() then both

1) sess.run(tf.global_variables_initializer())
2) tf.global_variables_initializer().run()

works to initialize variables.
"""

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

"""
tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the
model's unnormalized model prediction and sums across all classes, and tf.reduce_mean
takes the average over these sums.
"""
cross_entropy = tf.reduce_mean( \
                  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(2000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
