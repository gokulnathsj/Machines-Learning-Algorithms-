# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:37:15 2020

@author: USER
"""

import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values


#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y);

# fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# Predicting the new result with Polynomical Regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


# Visualising Regression results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or Bluff Polynomial Regression")
plt.xlabel("Postion Level")
plt.ylabel("Salary")
plt.show()

# Visualising Regression results with smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='red')
plt.title("Truth or Bluff Polynomial Regression")
plt.xlabel("Postion Level")
plt.ylabel("Salary")
plt.show()