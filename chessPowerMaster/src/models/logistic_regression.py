import tensorflow as tf
import numpy as np
from data_parser import get_binary_labeled_data
from model_evaluation import get_accuracy, evaluate_model
import time
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

class_number = 2
width = 50
height = 50
channels = 3
batch_size = 50
epochs = 50
display_step = 5

session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 1},
    allow_soft_placement=True,
    log_device_placement=True
)

tf.reset_default_graph()

# Getting data
train_images, train_labels, test_images, test_labels = get_binary_labeled_data(0.8, one_hot=True)

train_images = np.reshape(train_images, [-1, width * height * channels])
test_images = np.reshape(test_images, [-1, width * height * channels])

train_images = train_images / 255.0
test_images = test_images / 255.0

train_size = train_images.shape[0]
batch_size = train_size if train_size < batch_size else batch_size 

# tf Graph Input
images = tf.placeholder(tf.float32, [None, width * height * channels])
labels = tf.placeholder(tf.int32, shape=([None, class_number]))

# Set model weights
W = tf.Variable(tf.random_normal([width * height * channels, class_number], stddev=0.1))
b = tf.Variable(tf.zeros([class_number]))

# A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
#logits = tf.contrib.layers.fully_connected(inputs=images, num_outputs=class_number)

with tf.device('/gpu:0'):
    # Construct model
    logits = tf.matmul(images, W) + b
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    loss = tf.reduce_mean(entropy)
#regularizer = tf.nn.l2_loss(W)
#loss = tf.reduce_mean(loss + 0.001 * regularizer)

# Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

with tf.device('/cpu:0'):
    # Evaluate the model
    preds = tf.nn.softmax(logits)
    
    # Start training
    start_time = time.time()
    
with tf.Session(config=session_conf) as sess:

    # Run the initializer
    tf.global_variables_initializer().run()

    # Training cycle
    for epoch in range(epochs):
        print("--- %s seconds ---" % (time.time() - start_time))
        total_batch = int(train_size/batch_size)
        avg_train_loss = 0;
        avg_train_accuracy = 0;
        
        for batch in range(total_batch):
            
            start = batch * batch_size
            end = start + batch_size
            
            batch_images = train_images[start:end, :]
            batch_labels = train_labels[start:end, :]
            
            _, train_loss = sess.run([optimizer, loss], feed_dict={images: batch_images, labels: batch_labels})
            train_accuracy = get_accuracy(sess, labels, preds, feed_dict={images: batch_images, labels: batch_labels})
            
            avg_train_loss += train_loss/total_batch
            avg_train_accuracy += train_accuracy/total_batch
            
        if (epoch + 1) % display_step == 0:
            test_accuracy = get_accuracy(sess, labels, preds, feed_dict={images: test_images, labels: test_labels})
            print("Epoch " + str(epoch + 1))
            print("   Training loss: {:.2f}".format(avg_train_loss))
            print("   Training accuracy: {:.2f} %".format(100 * avg_train_accuracy))
            print("   Test accuracy: {:.2f} %".format(100 * test_accuracy))
    
    evaluate_model(sess, labels, preds, feed_dict={images: test_images, labels: test_labels})

print("--- %s seconds ---" % (time.time() - start_time))
    