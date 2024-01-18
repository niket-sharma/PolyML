
#Workshop 10.2: Prediction of HDPE Melt Index Using Deep Neural Networks
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Loading  Data
df = pd.read_excel('HDPE_LG_Plant_Data.xlsx')

y = df.iloc[:,10]
X = df.iloc[:,1:10]



#Splitting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y)



#Data Normalization

sc_x = StandardScaler().fit(X_train)
X_train_std = sc_x.transform(X_train)
X_test_std = sc_x.transform(X_test)

y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)
sc_y = StandardScaler().fit(y_train)
y_train_std = sc_y.transform(y_train)
y_test_std = sc_y.transform(y_test)



#Training Deep Learning Model

model = Sequential()
model.add(Dense(64, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='linear'))
model.summary()


#Compiling Model and Prediction
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(X_train_std,y_train_std, epochs=50, batch_size=150, verbose=1, validation_split=0.2)

#Loss history curve
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Model prediction on test data

y_pred_test_std = model.predict(X_test_std)
y_pred_test = sc_y.inverse_transform(y_pred_test_std)
r2_score(y_test, y_pred_test)



