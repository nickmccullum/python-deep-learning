import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_data = pd.read_csv('FB_training_data.csv')
training_data = training_data.iloc[:, 1].values
