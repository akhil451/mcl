#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:42:36 2017

@author: akhil
"""


"""
Created on Mon Jun 19 12:40:09 2017

@author: akhil
"""

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
y = df.iloc[:, 4].values#dependent



# training and testing set 
from sklearn.cross_validation import train_test_split
x_train ,x_test ,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=0)


# scaling the data 
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x_train=scX.fit_transform(x_train)
x_test=scX.fit_transform(x_test)

# create our classfier here 
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)
classifier.fit(x_train,y_train)
#predict
y_pred=classifier.predict(x_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)




# plotting 
from matplotlib.colors import ListedColormap

X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random FOrest (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

