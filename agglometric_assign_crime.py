
'''
Q)Perform clustering for the crime data and 
identify the number of clusters formed and draw inferences.
Refer to crime_data.csv dataset.

'''


#importing all libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

####################################################
#import crime data csv file
df=pd.read_csv("c:/2-dataset/crime_data.csv")

#####################################################
#see first five elements
df.head()
'''
   Unnamed: 0  Murder  Assault  UrbanPop  Rape
0     Alabama    13.2      236        58  21.2
1      Alaska    10.0      263        48  44.5
2     Arizona     8.1      294        80  31.0
3    Arkansas     8.8      190        50  19.5
4  California     9.0      276        91  40.6


#by using df.head we can see that first five rows and all column of dataset
'''
#########################################################

#check data-points and features in dataset
df.shape
''' 
(50, 5)

#crime dataset have 50 rows and 5 columns
'''
############################################################

#check the column names in dataset
df.columns
'''
['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'],
      
'''
#############################################################

#see five number summary of crime dataset
x=df.describe()
x
'''
 Murder     Assault   UrbanPop       Rape
count  50.00000   50.000000  50.000000  50.000000
mean    7.78800  170.760000  65.540000  21.232000
std     4.35551   83.337661  14.474763   9.366385
min     0.80000   45.000000  32.000000   7.300000
25%     4.07500  109.000000  54.500000  15.075000
50%     7.25000  159.000000  66.000000  20.100000
75%    11.25000  249.000000  77.750000  26.175000
max    17.40000  337.000000  91.000000  46.000000


#it describe the mean,standard deviation,min and max values,and also first quantile,second quantile and third quantile value.

'''
##########################################################
#Cumulative Distribution Function (CDF)
import numpy as np 
counts, bin_edges = np.histogram(df['Rape'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


################################################################

#Mean
print("Means:")
print(np.mean(df["Murder"]))
print(np.mean(df["Assault"]))
print(np.mean(df["UrbanPop"]))
print(np.mean(df["Rape"]))
'''
Means:
7.788
170.76
65.54
21.231
'''
##############################################################
#Median
print("Median:")
print(np.median(df["Murder"]))
print(np.median(df["Assault"]))
print(np.median(df["UrbanPop"]))
print(np.median(df["Rape"]))
'''
Median:
7.25
159.0
66.0
20.1
'''
############################################################

#Standard deviation
print("Standard deviation:")
print(np.std(df["Assault"]))
print(np.std(df["Assault"]))
print(np.std(df["UrbanPop"]))
print(np.std(df["Rape"]))
'''
Standard deviation:
82.50007515148094
82.50007515148094
14.329284699523559
9.272247623958283
'''
#################################################################

#we only required numerical data so delete nominal data column
y=df.drop(["Unnamed: 0"],axis=1)
y
'''here we drop unnamed:0 column because it contain nominal data'''

################################################################
#check datatypes of each column
y.dtypes
'''
Murder      float64
Assault       int64
UrbanPop      int64
Rape        float64

#here murder and rape columns are in float so convert it
'''
#############################################################
#change datatype of columns
y["Murder"]=y["Murder"].astype(int)
y["Rape"]=y["Rape"].astype(int)

y.dtypes
'''
Murder      int32
Assault     int64
UrbanPop    int64
Rape        int32

#all columns are converted into integer
'''
############################################################

#checking null values
y.isna().sum()
'''
Murder      0
Assault     0
UrbanPop    0
Rape        0

#none of column contain null value
'''
#############################################################
#checking any duplicate value is present or not
y.duplicated().sum()
'''it does not contain any duplicate value'''

###############################################################

#check outliers of all columns
sns.boxplot(y["Murder"])

sns.boxplot(y["Assault"])

sns.boxplot(y["UrbanPop"])

sns.boxplot(y["Rape"])
'''# rape column contains outliers '''

###########################################################

#remove outlier of rape column using winsorization
#winsorization is best method for removing outliers
#so import winsorizer model
from feature_engine.outliers import Winsorizer


#fit winsorizer model to rape column to remove outliers
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Rape"])
df_t=winsor.fit_transform(y[["Rape"]])
sns.boxplot(df_t)

'''#here outliers of rape column are get removed'''

#####################################################################

#check varience 
y.var()==0
'''#here is no varience is present in any column'''

################################################################
#apply normalization function for normalize the data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min()) #x=(a-b)/(z-b)
    return x

#apply that fun to all rows and columns of crime data
df_norm=norm_fun(y.iloc[:,:])
df_norm.describe()
 
###############################################################
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

#################################################################

#do hierarchical or aggomolative clustering on crime data
from sklearn.cluster import AgglomerativeClustering

h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(df_norm)

#apply labels to clusters
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

#assign this series to y["clust"] variable
y["clust"]=cluster_labels

###############################################################################
#select all rows and all column by their index
x=y.iloc[:,:]

#calculate mean and add clust column into it
x.iloc[:,:].groupby(x.clust).mean()
'''
          Murder     Assault   UrbanPop       Rape
clust                                             
0      12.052632  259.315789  68.315789  28.842105
1       5.800000  140.400000  70.350000  18.650000
2       2.363636   73.000000  52.000000  10.909091
'''

##########################################################
# generate crime_data csv file after clustering
x.to_csv("crime_data.csv",encoding="utf-8")

#display first five rows and all columns of file which is generated after clustering
x.head()
'''
   Murder  Assault  UrbanPop  Rape  clust
0      13      236        58    21      0
1      10      263        48    44      0
2       8      294        80    31      0
3       8      190        50    19      1
4       9      276        91    40      0
'''


#############################################################






