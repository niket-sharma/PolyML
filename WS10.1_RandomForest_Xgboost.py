#Workshop 10.1 : Prediction of HDPE Melt Index Using Random Forest and Extreme Gradient Boosting (XGBoost) Ensemble Learning Models

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


#Loading  Data
df = pd.read_excel('HDPE_LG_Plant_Data.xlsx')

y = df.iloc[:,10]
X = df.iloc[:,1:10]



#Data Visualization

p = df.iloc[:,1:11]
p.head()

plt.style.use('fivethirtyeight')
p.plot(subplots=True,
        layout=(6, 3),
        figsize=(22,22),
        fontsize=10, 
        linewidth=2,
        sharex=False,
        title='Visualization of the HDPE plant data')
plt.show()

#Dropping missing observations
df.dropna()

#Feature Selection

corr =X.corr()
import seaborn as sns
sns.heatmap(corr)

#Dropping highly correlated features in python

correlation_matrix = X.corr().abs()
correlated_features = set()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.95:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

len(correlated_features)

from sklearn.preprocessing import MinMaxScaler



#Splitting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y)


#Data Normalization
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler().fit(X_train)
X_train_std = sc_x.transform(X_train)
X_test_std = sc_x.transform(X_test)

y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)
sc_y = StandardScaler().fit(y_train)
y_train_std = sc_y.transform(y_train)
y_test_std = sc_y.transform(y_test)

n_train = np.count_nonzero(y_train)
y_train_std = y_train_std.reshape(n_train,)


#Training Random Forest Model 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)


rf.fit(X_train_std, y_train_std)

cv = cross_val_score(rf, X_train_std, y_train_std, cv = 10,scoring='neg_mean_squared_error') 
cv_score = cv.mean()

rmse_train = np.sqrt(abs(cv_score))
print(rmse_train)

#Training RMSE
y_rf_train_std = rf.predict(X_train_std)
y_rf_train_std = y_rf_train_std.reshape(-1,1)
y_rf_train = sc_y.inverse_transform(y_rf_train_std)

rmse_train = np.sqrt(mean_squared_error(y_train, y_rf_train))
print(rmse_train)

from sklearn.metrics import r2_score
r2_score(y_train, y_rf_train)

#TEST rmse
y_rf_std = rf.predict(X_test_std)
y_rf_std = y_rf_std.reshape(-1,1)
y_rf = sc_y.inverse_transform(y_rf_std)

rmse_test = np.sqrt(mean_squared_error(y_test, y_rf))
print(rmse_test)

#Hyperparameter tuning of Random Forest Model 
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_

best_grid = grid_search.best_estimator_

Y_best = best_grid.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y1_test, Y_best))
print(rmse)

#Feature Importance of Random Forest Model 

import time
import numpy as np

start_time = time.time()
importances = rf.feature_importances_
std = np.std([
    tree.feature_importances_ for tree in rf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")

feature_names = [f'feature {i}' for i in range(X.shape[1])]

import pandas as pd
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# Plotting  of plant data vs model prediction
import matplotlib.pyplot as plt
plt.style.use('default')
plt.scatter(t,y.iloc[:,1] ,c='green', label = 'Actual Density Estimate' )
plt.plot(t,Y_p[:,1] ,c = 'red', label = 'Predicted Density')
plt.xlabel('time')
plt.ylabel('Melt Index')

plt.legend()
plt.show()


#xgboost

#Model Training
import xgboost as xgb
xgr =xgb.XGBRFRegressor()
xgr.fit(X_train_std,y_train_std)

#Model Prediction
y_xgr_std = xgr.predict(X_test_std)
y_xgr_std = y_xgr_std.reshape(-1,1)
y_xgr = sc_y.inverse_transform(y_xgr_std)
y_xgr = xgr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_xgr))
print(rmse)


xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()