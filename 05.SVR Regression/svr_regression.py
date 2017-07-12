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
x= df.iloc[:,1:2].values#independent .[:, :1:2] so that X is a matrix , not a vector.
y = df.iloc[:,2:3].values#dependent

# scaling the data 
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x=scX.fit_transform(x)
#x_test=scX.fit(x_test)
scY=StandardScaler()
y=scY.fit_transform(y)


#fittng regression ,creating Regressor 

from sklearn.svm import SVR
regressor =SVR(kernel = 'rbf')
regressor.fit(x,y)



# prediction 
y_pred_p=scY.inverse_transform(regressor.predict(scX.transform(np.array([6.5]))))




#plotting with higher resolution
xp=np.linspace(min(x),max(x),100)
#xp = np.arange(min(X),max(X),0.1)
xp=xp.reshape(len(xp),1)
plt.scatter(x,y,c="r")
plt.plot(xp,regressor.predict(xp),c="b")
plt.show()

