# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:53:31 2020

@author: USER
"""

import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values


#feature Scaling
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train);'''

# Predicting the Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting the new result with Polynomical Regression
y_pred = regressor.predict([[6.5]])


# Visualising Regression results with smoother curve
X_grid = np.arange(X.min(), X.max(), .01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff Random Forest Regression")
plt.xlabel("Postion Level")
plt.ylabel("Salary")
plt.show()