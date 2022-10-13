# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:51:29 2022

@author: manik
"""

#CLUSTERING AIRLINES

import pandas as pd 
import numpy as np 
df=pd.read_excel("EastWestAirlines.xlsx",sheet_name="data")
df
df.shape       
df.head()     
df.info()     
df.isnull().sum()   

X= df.iloc[:, 1:12].values 
X
X= pd.DataFrame(df)
X.dtypes

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_scale = scale.fit_transform(X)
X_scale



#==================================================
# kmeans clustering


# by plotting elbow method we can decide which k is best
from sklearn.cluster import KMeans

inertia = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(X_scale)
    inertia.append(km.inertia_)
    
import matplotlib.pyplot as plt
%matplotlib qt
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()



KMeans = KMeans(n_clusters = 8, n_init=30)
KMeans.fit(X_scale)
Y = KMeans.predict(X_scale)
Y = pd.DataFrame(Y)
Y[0].value_counts()

C = KMeans.cluster_centers_
KMeans.inertia_








# CLUSTERING HIERARCHY 

import scipy.cluster.hierarchy as shc
# construction of Dendogram
#%matplotlib qt
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 20))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='average')) 


# Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
Y = cluster.fit_predict(X_scale)
Y = pd.DataFrame(Y)
Y[0].value_counts()

# DB SCAN
from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=3, min_samples=7)
dbscan.fit(X)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
cl['cluster'].value_counts()


clustered = pd.concat([df,cl],axis=1)

noisedata = clustered[clustered['cluster']==-1]
finaldata = clustered[clustered['cluster']==0]

clustered




clustered.mean()
finaldata.mean()









