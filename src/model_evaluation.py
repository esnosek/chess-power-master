import tensorflow as tf

def get_accuracy(sess, labels, preds, feed_dict):
    accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
    tf.local_variables_initializer().run()
    _, train_acc = sess.run([accuracy, accuracy_op], feed_dict)
    return train_acc

def get_precision(sess, labels, preds, feed_dict):
    precision, precision_op = tf.metrics.precision(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
    tf.local_variables_initializer().run()
    _, prec = sess.run([precision, precision_op], feed_dict)
    return prec

def get_recall(sess, labels, preds, feed_dict):
    recall, recall_op = tf.metrics.recall(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
    tf.local_variables_initializer().run()
    _, rec = sess.run([recall, recall_op], feed_dict)
    return rec

def get_false_positives(sess, labels, preds, feed_dict):
    false_positives, false_positives_op = tf.metrics.false_positives(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
    tf.local_variables_initializer().run()
    _, fp = sess.run([false_positives, false_positives_op], feed_dict)
    return fp

def get_false_negatives(sess, labels, preds, feed_dict):
    false_negatives, false_negatives_op = tf.metrics.false_negatives(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
    tf.local_variables_initializer().run()
    _, fn = sess.run([false_negatives, false_negatives_op], feed_dict)
    return fn

def get_true_positives(sess, labels, preds, feed_dict):
    true_positives, true_positives_op = tf.metrics.true_positives(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
    tf.local_variables_initializer().run()
    _, tp = sess.run([true_positives, true_positives_op], feed_dict)
    return tp

def get_true_negatives(sess, labels, preds, feed_dict):
    true_negatives, true_negatives_op = tf.metrics.true_negatives(labels=tf.argmax(labels, 1), predictions=tf.argmax(preds, 1))
    tf.local_variables_initializer().run()
    _, tn = sess.run([true_negatives, true_negatives_op], feed_dict)
    return tn

def evaluate_model(sess, labels, preds, feed_dict):
    test_acc = get_accuracy(sess, labels, preds, feed_dict)
    rec = get_recall(sess, labels, preds, feed_dict)
    prec = get_precision(sess, labels, preds, feed_dict)

    print("\nFinal train accuracy: {:.2f} %".format(100 * test_acc))
    print("Final recall: {:.2f} %".format(100 * rec))
    print("Final precision: {:.2f} %".format(100 * prec))
    
    fp = get_false_positives(sess, labels, preds, feed_dict)
    fn = get_false_negatives(sess, labels, preds, feed_dict)
    tp = get_true_positives(sess, labels, preds, feed_dict)
    tn = get_true_negatives(sess, labels, preds, feed_dict)
    
    print("False_positives: " + str(int(fp)))
    print("False_negatives: " + str(int(fn)))
    print("True_positives: " + str(int(tp)))
    print("True_negatives: " + str(int(tn)))