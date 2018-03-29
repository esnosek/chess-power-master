import tensorflow as tf
from data_parser import get_binary_labeled_data

class_number = 2
width = 200
height = 200
channels = 3
training_epochs = 1000
display_step = 1

X = tf.placeholder(tf.float32, [None, width, height, channels])
Y = tf.placeholder(tf.float32, [None, class_number])
W = tf.Variable(tf.random_normal([width * height * channels, class_number], stddev=0.01))
b = tf.Variable(tf.zeros([1, class_number]))

XX = tf.reshape(X, [-1, width * height * channels])

logits = tf.matmul(XX, W) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    all_x, all_y = get_binary_labeled_data()
    size = all_x.shape[0]

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        sess.run([optimizer, loss], feed_dict={XX: all_x, Y:all_y})

    print("Optimization Finished!")

    total_correct_preds = 0
    _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={XX: all_x, Y:all_y})
    preds = tf.nn.softmax(logits_batch)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(all_y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    total_correct_preds += sess.run(accuracy)
    print("Accuracy {0}".format(total_correct_preds/size))
