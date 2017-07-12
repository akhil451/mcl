#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing Template
"""

import pandas as pd
import numpy as np
df=pd.read_csv('50_Startups.csv')

# Importing the dataset

X = df.iloc[:, :-1].values #independent
y = df.iloc[:,4].values#dependent

#take care of categorical Data
# df['State'].value_counts()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEncoder_x=LabelEncoder()
X[:,3]=labelEncoder_x.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()           

#Avoid dummy variable Trap -- Not necessary , python libraries will take care of it .
X=X[:,1:]

# training and testing set 
from sklearn.cross_validation import train_test_split
x_train ,x_test ,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)


# scaling the data 

'''

from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x_train=scX.fit_transform(x_train)
x_test=scX.fit(x_test) 
'''

#fitting the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting test data
y_pred = regressor.predict(x_test)

# preparation for backward elimination
import statsmodels.formula.api as sm
X=np.append(np.ones((50,1)).astype(int),values=X,axis=1)

#backward  elimination
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog= y , exog=X_opt).fit()
regressor_OLS.summary()

#1 removing 2nd coloumn
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog= y , exog=X_opt).fit()
regressor_OLS.summary()

#
X_opt = X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog= y , exog=X_opt).fit()
regressor_OLS.summary()

#
X_opt = X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog= y , exog=X_opt).fit()
regressor_OLS.summary()
#
X_opt = X[:,[0,3]]
regressor_OLS=sm.OLS(endog= y , exog=X_opt).fit()
regressor_OLS.summary()

''' From the summary of OLS--ORdinary Least Squares --> The Profit Solely depends on the 
R&D Research 
'''
 