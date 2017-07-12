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
x = df.iloc[:,1:2].values#independent .[:, :1:2] so that X is a matrix , not a vector.
y = df.iloc[:,2].values#dependent

'''
# scaling the data 
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x_train=scX.fit_transform(x_train)
x_test=scX.fit(x_test)
'''



#fittng regression 


#creating Regressor 




# prediction 
y_pred_p=regressor.predict(6.5)


#plottiing
plt.scatter(x,y,c="r")
plt.plot(x,lin_reg.predict(X),c="green")


#plotting with higher resolution
xp=np.linspace(min(x),max(x),100)
#xp = np.arange(min(X),max(X),0.1)
xp=xp.reshape(len(xp),1)
plt.scatter(x,y,c="r")
plt.plot(xp,lin_reg1.predict(regressor_poly.fit_transform(xp)),c="b")
plt.show()

