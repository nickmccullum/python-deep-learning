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

#Training the neural network
ann.fit(x_training_data, y_training_data, batch_size = 32, epochs = 100)

#Making predictions with the artificial neural network
ann.predict(scaler.transform([[1, 0, 0, 555, 1, 52, 4, 75000, 3, 0, 1, 65000]]))

#Generate predictions from our test data
predictions = ann.predict(x_test_data) > 0.5

#Generate a confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_data, predictions)

#Generate an accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test_data, predictions)
