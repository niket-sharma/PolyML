

#Workshop Support Vector Machine

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


#Loading  Data
df = pd.read_excel('HDPE_LG_Plant_Data.xlsx')

y = df.iloc[:,10]
X = df.iloc[:,1:10]



#Splitting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y)



from sklearn.svm import SVR

svr = SVR(kernel = 'rbf' )
svr.fit(X_train, y_train)

y_svr = svr.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,y_svr)))