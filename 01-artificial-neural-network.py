#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#Import the data set and store it in a pandas DataFrame
raw_data = pd.read_csv('bank_data.csv')
x_data = raw_data.iloc[:, 3:-1].values
y_data = raw_data.iloc[:, -1].values

#Handle categorical data (gender first and then geography)
    #The Gender column uses label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x_data[:, 2] = label_encoder.fit_transform(x_data[:, 2])

    #The Geography column uses One Hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x_data = np.array(transformer.fit_transform(x_data))

#Split the data set into training data and test data
from sklearn.model_selection import train_test_split
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.3)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_training_data = scaler.fit_transform(x_training_data)
x_test_data = scaler.fit_transform(x_test_data)

#Building The Neural Network
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) #First hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) #Second hidden layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid')) #Output layer

#Compiling the neural network
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
