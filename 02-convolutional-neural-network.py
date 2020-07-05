#Import the necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Preprocessing the training set
training_generator = ImageDataGenerator(
                        rescale = 1/255,
                        shear_range = 0.2,
                        zoom_range = 0.2,
                        horizontal_flip = True)

training_set = training_generator.flow_from_directory('training_data',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#Preprocessing the test set
test_generator = ImageDataGenerator(rescale = 1./255)
test_set = test_generator.flow_from_directory('test_data',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#Building the artificial neural network
cnn = tf.keras.models.Sequential()

#Adding the convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#Adding our max pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Adding another convolutional layer and max pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Adding Our flattening Layer
cnn.add(tf.keras.layers.Flatten())

#Adding our full connection layer
cnn.add(tf.keras.layers.Dense(units=128, activation='sigmoid'))

#Adding our output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Compiling our convolutional neural network
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training our convolutional neural network
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

#Prediction preprocessing
from tensorflow.keras.preprocessing import image
test_image_1 = image.load_img('predictions/cat_or_dog_1.jpg', target_size = (64, 64))
test_image_2 = image.load_img('predictions/cat_or_dog_2.jpg', target_size = (64, 64))

test_image_1 = image.img_to_array(test_image_1)
test_image_2 = image.img_to_array(test_image_2)

test_image_1 = np.expand_dims(test_image_1, axis = 0)
test_image_2 = np.expand_dims(test_image_2, axis = 0)

#Making predictions on our two isolated images
print(cnn.predict(test_image_1))
print(cnn.predict(test_image_2))

#Determining which number corresponds to each animal
training_set.class_indices
