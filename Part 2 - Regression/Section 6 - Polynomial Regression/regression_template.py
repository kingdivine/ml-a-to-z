# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fit regression model to dataset
#---create regressor here----

#predict a new result with poly reg
y_pred = regressor.predict(6.5)

#visualise regression
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff - Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



#https://elitedatascience.com/overfitting-in-machine-learning