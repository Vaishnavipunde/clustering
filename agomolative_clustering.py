# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 08:14:09 2023

@author: sai
"""

#when dataset is large then use agomolative clustering
#when dataset is small then use k-means clustering


import pandas as pd

import matplotlib.pyplot as plt

univ1=pd.read_excel("C:/2-dataset/University_Clustering.xlsx")

univ1.describe()

#we have column state which is not useful

Univ=univ1.drop(["State"],axis=1)


#we know that there is scale difference among the columns which we have to remove either by using normalization or standardization whenever there is mix data do normalization

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#now apply this normalization fun to univ dataframe for all the rows and columns from 1 until end
df_norm=norm_fun(Univ.iloc[:,1:])
     

b=df_norm.describe()


#bbefore you apply clustering you need to plot dendrogram 

from scipy.cluster.hierarchy import linkage

import  scipy.cluster.hierarchy as sch

#linkage fun give agomolative clustering

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8))
plt.title("hirarchical clustering dendrogram")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()


#whatever displayed in dendrogram is not clustering it is just showing number of possible clusters

from sklearn.cluster import AgglomerativeClustering

h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(df_norm)

#apply labels to clusters

h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

#assign this series to univ

Univ["clust"]=cluster_labels

univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]

univ1.iloc[:,2:].groupby(univ1.clust).mean()

univ1.to_csv("university.csv",encoding="utf-8")
import os
os.getcwd()










