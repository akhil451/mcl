import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read 
df=pd.read_csv('/home/akhil/Desktop/dataScience/linear regresion/Salary_Data.csv')
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
#train-test
from sklearn.cross_validation import train_test_split 
x_train ,x_test ,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting test
y_pred =regressor.predict(x_test)

# plotting training set 
plt.scatter(x_train,y_train,c='r')
plt.plot(x_train,regressor.predict(x_train),c='b')
plt.title('training data plot')
plt.xlabel('Experience in years')
plt.ylabel('salary')
plt.show()

# plotting test set 
plt.scatter(x_test,y_test ,c="g")
plt.plot(x_train,regressor.predict(x_train))
plt.title('test data plot')
plt.xlabel('Experience in years')
plt.ylabel('salary')
plt.show()