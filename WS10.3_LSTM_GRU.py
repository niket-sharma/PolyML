# Workshop 10.3 Prediction of Time-Dependent HDPE Melt Index Using Dynamic Deep Recurrent Neural Networks

import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow
tensorflow.random.set_seed(1)
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


df1 = pd.read_csv('/content/drive/MyDrive/PolyData/HDPE_LG_Plant_Data.csv')
df1.head()

 X = df1.iloc[:,0:9]
 y = df1.iloc[:,9]

 from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler().fit(X)
X = sc_x.transform(X)
y = y.values.reshape(-1,1)
sc_y = StandardScaler().fit(y)
y = sc_y.transform(y)

# Defining Test, Train and Validation dataset 
X_train = X[:1500]
y_train = y[:1500]
X_val = X[1500:1800]
y_val = y[1500:1800]
X_test = X[1800:2400]
y_test = X[1800:2400]

# Reshape into 3D for LSTM input 
X_train = X_train.reshape(( X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape(( X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

#LSTM Model 
model = Sequential()
model.add(LSTM(units = 64,return_sequences = True,input_shape = (1,9)))
model.add(Dropout(0.25))
model.add(LSTM(units = 32,return_sequences = True))
model.add(Dropout(0.25))
model.add(Dense(1))


#GRU Model 
model = Sequential()
model.add(GRU(units = 64,return_sequences = True,input_shape = (1,9)))
model.add(Dropout(0.25))
model.add(GRU(units = 32,return_sequences = True))
model.add(Dropout(0.25))
model.add(Dense(1))


model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

#Model predciton
y_pred_std = model.predict(Xest)
y_pred = sc_y.inverse_transform(y_pred_std)
test_predictions = y_pred.flatten()
y_test = sc_y.inverse_transform(y_test)
y_test = y_test.flatten()

test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
test_results

import matplotlib.pyplot as plt
plt.plot(test_results['Test Predictions'][1800:2400],c ='green',label = 'LSTM prediction')
plt.plot(test_results['Actuals'][1800:2400],c ='red', label = 'Actual')
plt.xlabel('Time')
plt.ylabel('Melt Index')
plt.legend()
