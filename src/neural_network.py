import tensorflow as tf
from data_parser import get_binary_labeled_data

class_number = 2
width = 200
height = 200
channels = 3
features = width * height * channels
batch_size = 100
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

# Set model weights
L1 = 4000
L2 = 2000
L3 = 1000
L4 = 500

W1 = tf.Variable(tf.truncated_normal([features, L1], stddev=0.1))
b1 = tf.Variable(tf.zeros([L1]))
 
W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
b2 = tf.Variable(tf.zeros([L2]))

W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
b3 = tf.Variable(tf.zeros([L3]))

W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
b4 = tf.Variable(tf.zeros([L4]))

W5 = tf.Variable(tf.truncated_normal([L4, class_number], stddev=0.1))
b5 = tf.Variable(tf.zeros([class_number]))

# A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
#logits = tf.contrib.layers.fully_connected(inputs=images, num_outputs=class_number)

# Construct model
Y1 = tf.nn.sigmoid(tf.matmul(images, W1) + b1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + b2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + b3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + b4)
Ylogits = tf.matmul(Y4, W5) + b5

#logits = tf.matmul(images, W) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=labels)
loss = tf.reduce_mean(entropy)
#regularizer = tf.nn.l2_loss(W)
#loss = tf.reduce_mean(loss + 0.001 * regularizer)

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

# Evaluate the model
preds = tf.nn.softmax(Ylogits)
prediction_correct = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))

# Accuracy calculation
accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))
  
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
            
            _, train_loss, train_accuracy = sess.run([optimizer, loss, accuracy], feed_dict={images: batch_images, labels: batch_labels})
            
            avg_train_loss += train_loss/total_batch
            avg_train_accuracy += train_accuracy/total_batch
            
        if (epoch + 1) % display_step == 0:
            test_accuracy = sess.run(accuracy, feed_dict={images: test_images, labels: test_labels})
            print("Epoch " + str(epoch + 1))
            print("   Training loss: {:.2f}".format(avg_train_loss))
            print("   Training accuracy: {:.2f} %".format(100 * avg_train_accuracy))
            print("   Test accuracy: {:.2f} %".format(100 * test_accuracy))

    # Evaluation of the model
    test_accuracy = sess.run(accuracy, feed_dict={images: test_images, labels: test_labels})
    print("Final test accuracy: {:.2f} %".format(100 * test_accuracy))
