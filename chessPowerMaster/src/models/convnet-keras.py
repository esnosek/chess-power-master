import tensorflow as tf
from data_parser import get_binary_labeled_data

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y)
        if logs["acc"] == 1.0 and acc == 1.0:
            print('\nReached 100% accuracy for test and training data so cancelling training!')
            self.model.stop_training = True
        else:
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
  
 
class_number = 2
width = 50
height = 50
channels = 3

training_images, training_labels, test_images, test_labels = get_binary_labeled_data(
        test=0.8, width=50, height=50, channels=3, class_number=2, one_hot=False)


train_size = training_images.shape[0]
training_images=training_images / 255.0

test_size = test_images.shape[0]
test_images=test_images / 255.0

callbacks = TestCallback((test_images, test_labels))   

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(width, height, channels)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(width, height, channels)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(width, height, channels)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(class_number, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
print(model.evaluate(test_images, test_labels))
print(model.evaluate(training_images, training_labels))