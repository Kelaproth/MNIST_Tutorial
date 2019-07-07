import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create session
sess = tf.Session()

## reload trained model
# open saved model
saved = tf.train.import_meta_graph('./model/model.ckpt.meta')
saved.restore(sess, tf.train.latest_checkpoint('./model'))
# print(sess.graph.get_all_collection_keys())
# print(sess.graph.get_collection('train_op'))
# print(sess.graph.get_collection('trainable_variables'))
# print(sess.graph.get_collection('variables'))

# reload graph
graph = tf.get_default_graph()

# reload placeholder
x = graph.get_tensor_by_name('x:0')
y_ = graph.get_tensor_by_name('y_:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

# reload trained model and accuracy tensor
accuracy = graph.get_tensor_by_name('accuracy:0')
y = graph.get_tensor_by_name('y:0')


# print the predicted and actual digit
def drawDigit(position, image, title, right):
    # *(a, b) divide the tuple or list to separate arguments
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
    plt.axis('off')
    if not right:
        plt.title(title, color='red')
    else:
        plt.title(title)


# print some test image
def accuracy_graph():
    # random select 100 images and labels and form a batch
    images, labels = mnist.test.next_batch(100)
    # test the model
    predict_labels = sess.run(y, feed_dict={x: images, y_: labels, keep_prob: 1.0})
    # draw the graph
    image_number = images.shape[0]
    plt.figure(figsize=(18, 18))
    for i in range(10):
        for j in range(10):
            index = i * 10 + j
            if index < image_number:
                position = (10, 10, index+1)
                image = images[index]
                actual = np.argmax(labels[index])
                predict = np.argmax(predict_labels[index])
                right = (actual == predict)
                title = 'actual:%d\npredict:%d' % (actual, predict)
                drawDigit(position, image, title, right)


# Total accuracy
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print('Test accuracy: ', test_accuracy)
print("Loading some test result...")
accuracy_graph()
plt.show()

file_writer = tf.summary.FileWriter('./log_dir', sess.graph)

