from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# None means that a dimension can be of any length
x = tf.placeholder(tf.float32, [None, 784])

# Variable is a modifiable tensor
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
First we multiply x by W with the expression tf.matmul(x, W).

This is flipped from when we multiplied them in our equation, where we had
Wx,  as a smaill trick to deal with x being a 2D tensor with multiple inputs.

We then add b, and finally apply tf.nn.softmax.
"""
y = tf.nn.softmax(tf.matmul(x, W) + b)


"""
A function to determine the loss of a model is called "cross-entropy."
Cross-entropy arises from thinking about information compressing codes in
information theory.

To implement cross-entropy we need to first add a new placeholder to input
the correct answers.

Then we can implement the cross-entropy function.

"""
y_ = tf.placeholder(tf.float32, [None, 10])

"""
tf.reduce_sum adds the elements in the second dimension of y due to the
reduction_indice=[1] parameter. Finally tf.reduce_mean computes
the mean over all the examples in the batch.
"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# We ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""
We now launch the model in an Interactive Session.

The only difference with a regualr session is that an InteractiveSession installs
itself as the default session on construction.
"""
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

"""
Each step of the loop, we get a batch of one hundred random data points from
our training set.

Using small batches of random data is called stochastic training -- in this case,
stochastic gradient descent.

Ideally, we'd like to use all our data for every step of training because
that would give us a better sense of what we should be doing, but that's costly.
So, instead, we use a different subset every time. Doing this is cheap and
has much of the same benefit.
"""
for i in range(100):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, {x: batch_xs, y_: batch_ys})


"""
tf.argmax(y,1) is the label our model thinks is most likely for each input, while
tf.argmax(y_, 1) is the correct label.

Use tf.equal to check if our prediction matches the truth, whcih gives us a
list of booleans. To determine what fraction are correct, we cast to floating
point numbers and then take the mean.
"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()