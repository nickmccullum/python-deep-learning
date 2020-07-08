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

#Importing our TensorFlow libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#Initializing our recurrent neural network
rnn = Sequential()

#Adding our first LSTM layer
rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))

#Perform some dropout regularization
rnn.add(Dropout(0.2))

#Adding three more LSTM layers with dropout regularization
for i in [True, True, False]:
    rnn.add(LSTM(units = 45, return_sequences = i))
    rnn.add(Dropout(0.2))

#(Original code for the three additional LSTM layers)
# rnn.add(LSTM(units = 45, return_sequences = True))
# rnn.add(Dropout(0.2))

# rnn.add(LSTM(units = 45, return_sequences = True))
# rnn.add(Dropout(0.2))

# rnn.add(LSTM(units = 45))
# rnn.add(Dropout(0.2))

#Adding our output layer
rnn.add(Dense(units = 1))

#Compiling the recurrent neural network
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training the recurrent neural network
rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 32)
