#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 15:46:18 2017

@author: akhil
"""

import pandas as pd
df=pd.read_csv('Data.csv')

# Importing the dataset
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
X

# transform categorical data 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)
labelEncoder_x=LabelEncoder()
X[:,0]=labelEncoder_x.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

# training and testing set 
from sklearn.cross_validation import train_test_split
x_train ,x_test ,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)

# scaling the data 
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x_train=scX.fit_transform(x_train)
x_test=scX.fit(x_test)

#