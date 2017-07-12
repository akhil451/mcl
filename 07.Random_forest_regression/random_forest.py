#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing Template
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Position_Salaries.csv')

# Importing the dataset
x = df.iloc[:,1:2].values#independent .[:, :1:2] so that X is a matrix , not a vector.
y = df.iloc[:,2].values#dependent

'''
# scaling the data 
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x_train=scX.fit_transform(x_train)
x_test=scX.fit(x_test)
'''



# creating regressor and fitting it.
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(x,y)



# prediction 
y_pred_p=regressor.predict(6.5)

'''
normal plotting cannot be used since decision trees take the average of the decision set .
plotting has to be done in higher resolutions .

#plottiing
plt.scatter(x,y,c="r")
plt.plot(x,lin_reg.predict(X),c="green")
'''

#plotting with higher resolution
xp=np.linspace(min(x),max(x),1000)
#xp = np.arange(min(X),max(X),0.1)
xp=xp.reshape(len(xp),1)
plt.scatter(x,y,c="r")
plt.plot(xp,regressor.predict(xp),c="b")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("random forest Regression")
plt.show()

