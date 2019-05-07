import tensorflow as tf
from data_parser import get_binary_labeled_data
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

#callbacks = TestCallback((test_images, test_labels))   

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(width, height, channels)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(width, height, channels)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(width, height, channels)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
    

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              # optimizer="rmsprop",
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen  = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        '../data/train_data/empty_or_occupied_50x50/train',
        target_size=(width, height),
        batch_size=32,
        class_mode='binary')

validation_generator =  test_datagen.flow_from_directory(
        '../data/train_data/empty_or_occupied_50x50/validation',
        target_size=(width, height),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      validation_data=validation_generator,
      steps_per_epoch=16,  
      epochs=30,
      verbose=1)
