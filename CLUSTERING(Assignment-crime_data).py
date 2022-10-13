# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 23:50:46 2022

@author: manik
"""

# K means clustering
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df= pd.read_csv("crime_data.csv")
df.shape
list(df)
df.head()
df = df.iloc[:,2:]
df.shape

# standardization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_df_df=scaler.fit_transform(df.iloc[:,0:4])
scaled_df_df


# %matplotlib qt 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3],df.iloc[:,4])
plt.show()

#
from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters = 5, n_init=30)
Kmeans.fit(df)
Y = Kmeans.predict(df)

C = Kmeans.cluster_centers_
C[:,0]
C[:,1]
C[:,2]
#C[:,3]
#C[:,4]

Kmeans.inertia_


#%matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])
ax.scatter(C[:,0],C[:,1],C[:,2], marker='*',c='Red',s=1000)
plt.show()


# by plotting elbow method we can decide which k is best
from sklearn.cluster import KMeans

inertia = []
for i in range(1, 4):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(df)
    inertia.append(km.inertia_)
    
plt.plot(range(1, 4), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()


# DB scan clustering

#Import the libraries
import pandas as pd

# Import .csv file and convert it to a DataFrame object

array=df.values
array

from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X


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

#-----------------------------------------------------------------------

# Hierarchal clustering

import pandas as pd  
#import numpy as np  

customer_data = pd.read_csv("crime_data.csv", delimiter=',') 
customer_data.shape
customer_data.head()
X = customer_data.iloc[:, 1:5].values 
X.shape

##############################################################################

import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='average')) 


## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')
Y = cluster.fit_predict(X)

Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(X[:,1], X[:,2], c=cluster.labels_, cmap='rainbow')  

Y_clust = pd.DataFrame(Y)
Y_clust[0].value_counts()
##############################################################################

