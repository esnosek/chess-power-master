import tensorflow as tf
from data_parser import get_binary_labeled_data
from model_evaluation import get_accuracy, evaluate_model

class_number = 2
width = 50
height = 50
channels = 3
features = width * height * channels
batch_size = 500
epochs = 1000
display_step = 1

tf.reset_default_graph()

# Getting data
train_images, train_labels, test_images, test_labels = get_binary_labeled_data(0.8, one_hot=True)

train_size = train_images.shape[0]
batch_size = train_size if train_size < batch_size else batch_size 

# tf Graph Input
images = tf.placeholder(tf.float32, [None, features])
labels = tf.placeholder(tf.int32, shape=([None, class_number]))

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
        inputs=images,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 
conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

# Pooling Layer #1
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#logits = tf.matmul(images, W) + b

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=labels)
loss = tf.reduce_mean(entropy)
#regularizer = tf.nn.l2_loss(W)
#loss = tf.reduce_mean(loss + 0.001 * regularizer)

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Evaluate the model
preds = tf.nn.softmax(Ylogits)
  
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(epochs):
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
