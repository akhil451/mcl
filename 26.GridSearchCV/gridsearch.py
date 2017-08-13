

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing Template
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
df=pd.read_csv( 'Social_Network_Ads.csv')

# Importing the dataset
x = df.iloc[:,[2,3]].values#independent
y = df.iloc[:, -1].values#dependent



# training and testing set 
from sklearn.cross_validation import train_test_split
x_train ,x_test ,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=0)


# scaling the data 
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x_train=scX.fit_transform(x_train)
x_test=scX.fit_transform(x_test)

# create our classfier here 
from sklearn.svm  import SVC
classifier = SVC(kernel="rbf",random_state=0)
classifier.fit(x_train,y_train)


#predict
#y_pred=classifier.predict(x_test)



#Kmean
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100],'kernel':['linear']},
            {'C':[1,10,100],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
           ]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,cv=10,scoring='accuracy',n_jobs=-1)
grid_search=grid_search.fit(x_train,y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
            #plotting 
