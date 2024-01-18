# -*- coding: utf-8 -*-
# Workshop 10.5: Melt Index Prediction Using Automated Machine Learning

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

from google.colab import drive
drive.mount('/content/drive')

import sys
# modify "customized_path_to_homework", path of folder in drive, where you uploaded your homework
customized_path = "/content/drive/My Drive/PolyData"
sys.path.append(customized_path)

ls /content/drive/"My Drive"/PolyData

df1 = pd.read_csv('/content/drive/MyDrive/PolyData/HDPE_LG_Plant_Data.csv')
df1.head()



"""# New Section

# New Section
"""

X = df1.iloc[:,0:10]
 X.head()

y = df1.iloc[:,10]
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

pip install autosklearn

pip install h2O

import h2o
from h2o.automl import H2OAutoML

h2o.init()

df = h2o.import_file(path = '/content/drive/MyDrive/PolyData/HDPE_LG_Plant_Data.csv')

df.describe(chunk_summary=True)

train, test = df.split_frame(ratios=[0.8], seed = 1)

aml = H2OAutoML(max_models =25,
                balance_classes=True,
		seed =1)

train.head()

aml.train(training_frame = train, y = 'MI_Plant')

lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

preds = aml.predict(test)

preds = aml.leader.predict(test)

lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
lb

# Get the best model using the metric
m = aml.leader
# this is equivalent to
m = aml.get_best_model()

print(m)

aml.train(training_frame = train,
          y = 'MI_plant',
	  leaderboard_frame = lb)

best_model = aml.get_best_model()
print(best_model)

best_model.model_performance(test)

explain_model = aml.explain(frame = test, figsize = (8,6))

