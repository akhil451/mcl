
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
df=pd.read_csv('Churn_Modelling.csv')

# Importing the dataset
X = df.iloc[:, 3:13].values#independent
y = df.iloc[:, -1].values#dependent

# label encoding and one_hot_encoding for categorical Data 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_country=LabelEncoder()
le_gender=LabelEncoder()
X[:,1]=le_country.fit_transform(X[:,1])
X[:,2]=le_gender.fit_transform(X[:,2])
ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X=X[:,1:]
# training and testing set 
from sklearn.model_selection import train_test_split
x_train ,x_test ,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
x_train=scx.fit_transform(x_train)
x_test=scx.fit_transform(x_test)


# ANN Libraries 
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout
classifier=Sequential()

#1st hidden layer
classifier.add(Dense(6,activation='relu',init='uniform',input_dim=11))
classifier.add(Dropout(0.1))
#2nd hidden layer 
classifier.add((Dense(6,activation='relu',init='uniform')))
classifier.add(Dropout(0.1))
#output layer 
classifier.add(Dense(1,activation='sigmoid',init='uniform'))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#adam-->schocastic gradient descent 
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

y_pred=classifier.predict(x_test)
y_pred =(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
print(cm)
accuracy=((cm[0,0]+cm[1,1])/2000)
print('accuracy:',accuracy)




# predicting for 1 customer ...
y_pred1=classifier.predict(scx.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
y_pred1=(y_pred1>0.5)
print(y_pred1)
