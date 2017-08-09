#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:06:34 2017

@author: akhil
"""

import pandas as pd 

df= pd.read_csv('Wine.csv')
X=df.iloc[:,0:13].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
X_train=scx.fit_transform(X_train)
X_test=scx.transform(X_test)


from sklearn.decomposition import PCA
pca=PCA(n_components=3)
X_train = pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy=((cm[0][0]+cm[1][1]+cm[2][2])/len(y_pred) )*100
print('accuracy:',accuracy)


#plotting

import numpy as np
import seaborn as sns
sns.set(style="darkgrid")



