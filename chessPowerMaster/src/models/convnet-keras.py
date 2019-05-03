import tensorflow as tf
from data_parser import get_binary_labeled_data

class_number = 2
width = 50
height = 50
channels = 3

training_images, training_labels, test_images, test_labels = get_binary_labeled_data(0.8, one_hot=False)

train_size = training_images.shape[0]
training_images=training_images / 255.0

test_size = test_images.shape[0]
test_images=test_images / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(width, height, channels)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(class_number, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=20)
print(model.evaluate(test_images, test_labels))
print(model.evaluate(training_images, training_labels))