import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# import the database
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set the placeholder
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

# reshape the data
x_image = tf.reshape(x, [-1, 28, 28, 1])

## the first convolutional layer
# define the Weight and bias
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='W_conv1')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')
# We then convolve x_image with the weight tensor,
# add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, name='h_conv1')
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')

## the second convolutional layer
# This time, to got a deeper network,
# we will pile a few similar layers,
# in second layer, every 5*5 patch will get 64 characteristics
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='W_conv2')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2')

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, name='h_conv2')
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')

## the third layer, dense layer, or fully connected layer
# resize the image to 7*7, add a fully connected layer with 1024 nodes

# reshape the tensor to vectors
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')

W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name='W_fc1')
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')

## Dropout, to prevent over-fitting
# fc1 dropout (used for train p = 0.5, not used for test p =1.0)
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

## output layer, with softmax operation, also the second fully connected layer
W = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='W')
b = tf.Variable(tf.constant(0.1, shape=[10]), name='b')

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W) + b, name='y')  # predicted results

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)), name='cross_entropy')

# use a more complicated optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='train_step')

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# saver to save all variable
saver = tf.train.Saver()

# start training, 20000 times
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step {0}, training accuracy {1:.4f}".format(i, train_accuracy))

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

# save the trained model
save_path = saver.save(sess, "./model/model.ckpt")
print("Model saved in path: %s" % save_path)

# print("")
# test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
# print("test accuracy ", test_accuracy)

