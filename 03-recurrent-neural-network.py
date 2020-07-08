#Import the necessary data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the data set as a pandas DataFrame
training_data = pd.read_csv('FB_training_data.csv')

#Transform the data set into a NumPy array
training_data = training_data.iloc[:, 1].values

#Apply feature scaling to the data set
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data.reshape(-1, 1))

#Initialize our x_training_data and y_training_data variables 
#as empty Python lists
x_training_data = []
y_training_data =[]

#Populate the Python lists using 40 timesteps
for i in range(40, len(training_data)):
    x_training_data.append(training_data[i-40:i, 0])
    y_training_data.append(training_data[i, 0])
    
#Transforming our lists into NumPy arrays
x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)

#Verifying the shape of the NumPy arrays
print(x_training_data.shape)
print(y_training_data.shape)

#Reshaping the NumPy array to meet TensorFlow standards
x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], 
                                               x_training_data.shape[1], 
                                               1))

#Printing the new shape of x_training_data
print(x_training_data.shape)
