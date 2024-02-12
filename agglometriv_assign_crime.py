# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:25:18 2023

@author: sai
"""

#importing all libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#import crime data csv file
df=pd.read_csv("crime_data.csv")

#see first five elements
df.head()

x=df.describe()
x

#we only required numerical data so delete nominal data column
y=df.drop(["Unnamed: 0"],axis=1)
y

#check datatypes of each column
y.dtypes

#convert float value into int of murder column
y["Murder"]=y["Murder"].astype(int)

#convert float value into int of rape column
y["Rape"]=y["Rape"].astype(int)

#again check datatypes of columns
y.dtypes

#checking null values
y.isna().sum()

#checking any duplicate value is present or not
y.duplicated().sum()

#check outlier is present in murder column
sns.boxplot(y["Murder"])












