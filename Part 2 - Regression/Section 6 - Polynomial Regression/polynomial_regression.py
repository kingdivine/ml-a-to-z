#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:48:36 2020

@author: divine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#fit linear regression to dataset (doing this just for comparison)
#the model is clearly going to be better as a polynomial
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fit polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_reg.fit_transform(X), y)

#visualise linear
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff - Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#https://elitedatascience.com/overfitting-in-machine-learning

#visualise poly
X_grid = np.arange(min(X), max(X), 0.1) # to make graph more fine grained (curve will now be predicting 10 times as many values)
X_grid = X_grid.reshape(len(X_grid), 1) #turn from vector to matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or bluff - Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predict a new result with linear regression
lin_reg.predict([[6.5]])

#predict a new result with poly reg
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
