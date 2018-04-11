import tensorflow as tf
from data_parser import get_binary_labeled_data

class_number = 2
width = 200
height = 200
channels = 3
batch_size = 500
epochs = 2000
display_step = 100

tf.reset_default_graph()

# Getting data
train_images, train_labels, test_images, test_labels = get_binary_labeled_data(0.8, one_hot=True)

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

# Construct model
logits = tf.matmul(images, W) + b
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
loss = tf.reduce_mean(entropy)
#regularizer = tf.nn.l2_loss(W)
#loss = tf.reduce_mean(loss + 0.001 * regularizer)

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

# Evaluate the model
preds = tf.nn.softmax(logits)

prediction_correct = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
accuracy2 = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

# Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()

accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
recall, recall_op = tf.metrics.recall(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
precision, precision_op = tf.metrics.precision(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))

false_positives, false_positives_op = tf.metrics.false_positives(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
false_negatives, false_negatives_op = tf.metrics.false_negatives(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
true_positives, true_positives_op = tf.metrics.true_positives(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
true_negatives, true_negatives_op = tf.metrics.true_negatives(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))

# Start training
with tf.Session() as sess:

    # Run the initializer
    tf.global_variables_initializer().run()

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
            tf.local_variables_initializer().run()
            _, train_accuracy = sess.run([accuracy, accuracy_op], feed_dict={images: batch_images, labels: batch_labels})
            
            avg_train_loss += train_loss/total_batch
            avg_train_accuracy += train_accuracy/total_batch
            
        if (epoch + 1) % display_step == 0:
            tf.local_variables_initializer().run()
            _, test_accuracy = sess.run([accuracy, accuracy_op], feed_dict={images: test_images, labels: test_labels})
            print("Epoch " + str(epoch + 1))
            print("   Training loss: {:.2f}".format(avg_train_loss))
            print("   Training accuracy: {:.2f} %".format(100 * avg_train_accuracy))
            print("   Test accuracy: {:.2f} %".format(100 * test_accuracy))
    
    # Model eveluation
    
    tf.local_variables_initializer().run()
    _, train_acc = sess.run([accuracy, accuracy_op], feed_dict={images: train_images, labels: train_labels})
    tf.local_variables_initializer().run()
    _, acc = sess.run([accuracy, accuracy_op], feed_dict={images: test_images, labels: test_labels})
    tf.local_variables_initializer().run()
    _, rec = sess.run([recall, recall_op], feed_dict={images: test_images, labels: test_labels})
    tf.local_variables_initializer().run()
    _, prec = sess.run([precision, precision_op], feed_dict={images: test_images, labels: test_labels})

    print("\nFinal train accuracy: {:.2f} %".format(100 * train_acc))
    print("Final test accuracy: {:.2f} %".format(100 * acc))
    print("Final recall: {:.2f} %".format(100 * rec))
    print("Final precision: {:.2f} %".format(100 * prec))
    
    _, fp = sess.run([false_positives, false_positives_op], feed_dict={images: test_images, labels: test_labels})
    _, fn = sess.run([false_negatives, false_negatives_op], feed_dict={images: test_images, labels: test_labels})
    _, tp = sess.run([true_positives, true_positives_op], feed_dict={images: test_images, labels: test_labels})
    _, tn = sess.run([true_negatives, true_negatives_op], feed_dict={images: test_images, labels: test_labels})
    
    print("False_positives: " + str(int(fp)))
    print("False_negatives: " + str(int(fn)))
    print("True_positives: " + str(int(tp)))
    print("True_negatives: " + str(int(tn)))
    