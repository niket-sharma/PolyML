
#Workshop 10.4 Polymer Property Prediction Using Molecular Structure Using Convolutional Neural Network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import ColorConverter
import numpy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense,MaxPool2D,Conv2D,Flatten
from keras import optimizers
from keras.layers import Dropout

df=pd.read_csv("/content/drive/MyDrive/PolyData/Polymer_Tg_SMILES.csv",encoding='windows-1254')
df.head()


#One Hot encoding of SMILES data
d=[]
n=[['c'], ['n'], ['o'], ['C'], ['N'], ['F'], ['='], ['O'], 
            ['('], [')'], ['1'],['2'],['#'],['Cl'],['/'],['S'],['Br']]
e = OneHotEncoder(handle_unknown='ignore')
e.fit(n)
e.categories_
df1=df["SMILES Structure"].apply(lambda x: pd.Series(list(x)))
for i in range(df1.shape[0]):
    x=e.transform(pd.DataFrame(df1.iloc[i,:]).dropna(how="all").values).toarray()
    y=np.zeros(((df1.shape[1]-x.shape[0]),len(n)))
    d.append(np.vstack((x,y)))


# COnverting encoded SMILES to binary images
  plt.figure(figsize=(20,100))
  for i in range(len(d)):
    plt.subplot(len(d),5,i+1)
    plt.imshow(d[i])

#Dataset 
X = np.array(d)
Y=df["Tg"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=0)

#CNN Model 

model=Sequential()
model.add(Conv2D(128,(3,3), activation="relu",input_shape=(65,17,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization()),
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization()),
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])  
history=model.fit(x=X_train,y=y_train,epochs=500,batch_size=16,validation_split=0.1)

y_predtrain=model.predict(X_train)
y_predtest=model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_predtest.reshape(y_test.shape)))
print(rmse_test)

r2_score(y_test, y_predtest.reshape(y_test.shape))