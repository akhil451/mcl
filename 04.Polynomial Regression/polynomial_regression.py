#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing Template
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/home/akhil/Desktop/dataScience/Polynomial Regression/Position_Salaries.csv')

# Importing the dataset
X = df.iloc[:,1:2].values#independent .[:, :1:2] so that X is a matrix , not a vector.
y = df.iloc[:,2].values#dependent

'''
# training and testing set 
from sklearn.cross_validation import train_test_split
x_train ,x_test ,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)
'''
           
           
'''
# scaling the data 
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x_train=scX.fit_transform(x_train)
x_test=scX.fit(x_test)
'''


# fitting linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
lin_reg=regressor.fit(X,y)

#plottiing linear regression
plt.scatter(X,y,c="r")
plt.plot(X,lin_reg.predict(X),c="green")

# prediction salary from Linear REgreession
pred_l=lin_reg.predict(6.5)

#fittng polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
regressor_poly=PolynomialFeatures(degree=4)
x_poly=regressor_poly.fit_transform(X)
regressor_poly.fit(x_poly,y)
lin_reg1=LinearRegression()
lin_reg1.fit(x_poly,y)

#plotting the polynomial curve
xp=np.linspace(min(X),max(X),100)
#xp = np.arange(min(X),max(X),0.1)
xp=xp.reshape(len(xp),1)
plt.scatter(X,y,c="r")
plt.plot(xp,lin_reg1.predict(regressor_poly.fit_transform(xp)),c="b")
plt.show()

# prediction salary from Linear REgreession
pred_p=lin_reg1.predict(regressor_poly.fit_transform(6.5))

