#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:26:20 2017

@author: akhil
"""

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
sns.set()
df=pd.read_csv("Restaurant_Reviews.tsv",sep="\t",quoting=3)
# qouting=3 -- for ignoring the qoutes .
# https://stackoverflow.com/questions/43344241/quoting-parameter-in-pandas-read-csv


 # cleaning data
import nltk 
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
corpus=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]'," ",df["Review"][i])
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
    
    
# bag of words .

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()# scarce matrix
y=df.iloc[:,1].values


#training the model on naive bayes 
from sklearn.cross_validation import train_test_split
x_train ,x_test ,y_train,y_test =train_test_split(x,y,test_size=0.20,random_state=0)



# Naive Bayes
'''
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#predict
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
naive_bayes_correct=cm[0][0]+cm[1][1]
naive_bayes_wrong=cm[1][0]+cm[0][1]

accuracy=naive_bayes_correct/200
precision=cm[1][1]/200
'''



#Random Forest

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)
classifier.fit(x_train,y_train)
#predict
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
Random_forest_correct=cm[0][0]+cm[1][1]
Random_forest_wrong=cm[1][0]+cm[0][1]

accuracy=Random_forest_correct/200
precision=cm[1][1]/200




