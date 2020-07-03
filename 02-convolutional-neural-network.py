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

#Preprocessing the test set

