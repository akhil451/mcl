#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:34:17 2017

@author: akhil
"""

import pandas as pd 
import matplotlib.pyplot as plt 
df=pd.read_csv('Mall_Customers.csv')

X=df.iloc[:,[3,4]].values
from scipy.cluster.hierarchy import linkage,dendrogram 
dendrogram(linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distance(Euclidean)')
plt.show()
plt.savefig('Dendrogram.png')

''' from HC dendrogram n_clusters=5

Now Fitting the HC MOdel to the data

'''

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,linkage='ward')
y_hc=hc.fit_predict(X)


#plotting
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.show()
plt.savefig('Clusters.png')