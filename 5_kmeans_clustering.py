# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:26:56 2023

@author: sai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random numbers from 0 to 1 for X and Y coordinates
X = np.random.uniform(0, 1, 50)
Y = np.random.uniform(0, 1, 50)

# Create a DataFrame to store the generated data
df_xy = pd.DataFrame({"X": X, "Y": Y})

# Plot the generated data as a scatter plot
df_xy.plot(x="X", y="Y", kind="scatter")
plt.title('Randomly Generated Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Fit a KMeans clustering model to the data
model = KMeans(n_clusters=3).fit(df_xy)


#generate scatter plot with data x and y ,apply kmeans model with scale or font 10
#cmap.plt.cm.coolwarm is colour combination for scatter plot
model.labels_
# Generate scatter plot using the KMeans labels with customization
df_xy.plot(x="X", y="Y", c=model.labels_, kind="scatter", s=10, cmap=plt.cm.coolwarm)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with KMeans Labels')
plt.show()



univ1=pd.read_excel("C:/2-dataset/University_Clustering.xlsx")

univ1.describe()

# Drop the 'State' column as it's not useful for clustering
Univ = univ1.drop(["State"], axis=1)

# Normalization function
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

# Apply normalization to the University data except the first column (University names)
df_norm = norm_func(Univ.iloc[:, 1:])

# Initialize empty list for Total Within Sum of Squares (TWSS)
TWSS = []

# Iterate over different values of K (clusters)
for i in range(2, 8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    # Append inertia (Total Within Sum of Squares) to TWSS list
    TWSS.append(kmeans.inertia_)

# Plot Elbow curve
plt.plot(range(2, 8), TWSS, "ro-")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Total Within Sum of Squares (TWSS)")
plt.title("Elbow Curve for Optimal K")
plt.show()

# Selecting K=3 based on the Elbow curve analysis
# how to select value of k from elbow curve when k changes from 2 to 3 then decrease in twss is higher than when k changes from 3 to 4,when k value changes from 5 to  6 decreases in twss is considerably less hence k=3
# Fit KMeans model with selected number of clusters (K=3)
model = KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_

# Assign cluster labels to the original dataset
mb=pd.Series(model.labels_)
Univ['clust']=mb

Univ.head()

# Extract the mean values for each cluster for the numerical columns
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ=Univ.iloc[:,2:8].groupby(Univ.clust).mean()

# Save the cluster means to a CSV file
Univ.to_csv("kmeans.csv",encoding="utf-8")

#for getting path of file
import os
os.getcwd()







