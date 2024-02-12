# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:30:16 2023

@author: sai


#1.	Perform clustering for the airlines data to obtain 
#optimum number of clusters. Draw the inferences from 
#the clusters obtained. Refer to EastWestAirlines.xlsx
# dataset.


"""


#importing all libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

###############################################################
#import crime data csv file

df=pd.read_excel("c:/2-dataset/EastWestAirlines.xlsx")

#see first five elements
df.head()

x=df.describe()
x

df.columns


#Cumulative Distribution Function (CDF)
import numpy as np 
counts, bin_edges = np.histogram(df['Bonus_trans'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



#we only required numerical data so delete nominal data column
y=df.drop(["Balance","Award?"],axis=1)
y

#check datatypes of each column
y.dtypes

#checking null values
y.isna().sum()

#checking any duplicate value is present or not
y.duplicated().sum()

y.columns


sns.boxplot(y["ID#"])#n

sns.boxplot(y["Qual_miles"])

sns.boxplot(y["cc1_miles"])#n

sns.boxplot(y["cc2_miles"])

sns.boxplot(y["cc3_miles"])

sns.boxplot(y["Bonus_miles"])

sns.boxplot(y["Bonus_trans"])

sns.boxplot(y["Flight_miles_12mo"])

sns.boxplot(y["Flight_trans_12"])

sns.boxplot(y["Days_since_enroll"])#n



#let us calculate IQR
IQR=df.Qual_miles.quantile(0.75)-df.Qual_miles.quantile(0.25)
lower_limit=df.Qual_miles.quantile(0.25)-1.5*IQR
upper_limit=df.Qual_miles.quantile(0.75)+1.5*IQR
#replace outlier with upperlimit and lowerlimit
df_replaced=pd.DataFrame(np.where(df.Qual_miles>upper_limit,upper_limit,np.where(df.Qual_miles<lower_limit,lower_limit,df.Qual_miles)))
sns.boxplot(df_replaced[0])



#let us calculate IQR
IQR1=df.cc2_miles.quantile(0.75)-df.cc2_miles.quantile(0.25)
lower_limit=df.cc2_miles.quantile(0.25)-1.5*IQR1
upper_limit=df.cc2_miles.quantile(0.75)+1.5*IQR1
#replace outlier with upperlimit and lowerlimit
df_replaced1=pd.DataFrame(np.where(df.cc2_miles>upper_limit,upper_limit,np.where(df.cc2_miles<lower_limit,lower_limit,df.cc2_miles)))
sns.boxplot(df_replaced1[0])



#let us calculate IQR
IQR2=df.cc2_miles.quantile(0.75)-df.cc2_miles.quantile(0.25)
lower_limit=df.cc2_miles.quantile(0.25)-1.5*IQR2
upper_limit=df.cc2_miles.quantile(0.75)+1.5*IQR2
#replace outlier with upperlimit and lowerlimit
df_replaced2=pd.DataFrame(np.where(df.cc2_miles>upper_limit,upper_limit,np.where(df.cc2_miles<lower_limit,lower_limit,df.cc2_miles)))
sns.boxplot(df_replaced2[0])



#let us calculate IQR
IQR3=df.cc3_miles.quantile(0.75)-df.cc3_miles.quantile(0.25)
lower_limit=df.cc3_miles.quantile(0.25)-1.5*IQR3
upper_limit=df.cc3_miles.quantile(0.75)+1.5*IQR3
#replace outlier with upperlimit and lowerlimit
df_replaced3=pd.DataFrame(np.where(df.cc3_miles>upper_limit,upper_limit,np.where(df.cc3_miles<lower_limit,lower_limit,df.cc3_miles)))
sns.boxplot(df_replaced3[0])



#remove outlier of rape column using winsorization
#winsorization is best method for removing outliers
#so import winsorizer model
from feature_engine.outliers import Winsorizer

#fit winsorizer model to rape column to remove outliers


winsor3=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Bonus_miles"])
df_t3=winsor3.fit_transform(y[["Bonus_miles"]])
sns.boxplot(df_t3)#y

winsor4=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Bonus_trans"])
df_t4=winsor4.fit_transform(y[["Bonus_trans"]])
sns.boxplot(df_t4)#y


winsor5=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Flight_miles_12mo"])
df_t5=winsor5.fit_transform(y[["Flight_miles_12mo"]])
sns.boxplot(df_t5)#y


winsor6=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Flight_trans_12"])
df_t6=winsor6.fit_transform(y[["Flight_trans_12"]])
sns.boxplot(df_t6)#y


#here outliers of rape column are get removed

#check varience 
y.var()==0
#here is no varience is present in any column

#apply normalization function for normalize the data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min()) #x=(a-b)/(z-b)
    return x

#apply that fun to all rows and columns of crime data
df_norm=norm_fun(y.iloc[:,:])
df_norm.describe()
 
#import linkage model for plotting dendrogram
from scipy.cluster.hierarchy import linkage
import  scipy.cluster.hierarchy as sch

#plot the dendrogram for crime data
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8))
plt.title("hierarchical clustering dendrogram")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z)
plt.show()

#do hierarchical or aggomolative clustering on crime data
from sklearn.cluster import AgglomerativeClustering

h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(df_norm)

#apply labels to clusters

h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

#assign this series to x variable
y["clust"]=cluster_labels

x=y.iloc[:,:]

#calculate mean and add clust column into it
x.iloc[:,:].groupby(x.clust).mean()

# generate crime_data csv file after clustering
x.to_csv("Airlines.csv",encoding="utf-8")

z=x.head()










