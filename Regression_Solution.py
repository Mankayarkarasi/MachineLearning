# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:26:03 2017

@author: mankayarkarasi.c
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Correlation function for features

def correlation_heatmap(df, title):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(10, 240, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True,
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title(title, y=1.05, size=15)
    
    
#Importing the Big Mart Data Train
Big_Mart_Data=pd.read_csv("C:\\Users\\mankayarkarasi.c\\Desktop\\AIML\\Regression\\Train_Sales_Data.csv")

#Importing the Big Mart Data Test
Big_Mart_Data_test = pd.read_csv('C:\\Users\mankayarkarasi.c\\Desktop\\AIML\Regression\\Predict_Sales_Data.csv')

#===============================
#1. Data Exploration
#===============================

#Summary of data
Big_Mart_Data.describe()

#Structure of data
Big_Mart_Data.info()

#Moving to nominal (categorical) variable, lets have a look at the number of unique values in each of them.

Big_Mart_Data.apply(lambda x: len(x.unique()))

#Correlation plot
correlation_heatmap(Big_Mart_Data, 'Correlat')

# Create correlation matrix
corr_matrix = Big_Mart_Data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
Big_Mart_Data.drop(to_drop, axis=1,inplace=True)

#Boxplot for all the 12 variables to observe outliers.
Big_Mart_Data.plot(kind='box', figsize=[20,20])

#===============================
#2.Data Cleaning for Train
#===============================

# count the number of NaN values in each column
print(Big_Mart_Data.isnull().sum())

# Write a function that imputes mean(because numerical data)
def impute_mean(series):
    return series.fillna(series.mean())

# Impute NA Values of Outlet_Size & 'Item_Weight
def Big_Mart_impute(Data):
    Data['Item_Weight'] = impute_mean(Data['Item_Weight'])
    mode_value=Data['Outlet_Size'].mode()
    Data['Outlet_Size']=Data['Outlet_Size'].fillna(mode_value[0])

Big_Mart_impute(Big_Mart_Data)
# count the number of NaN values in each column after Imputation
print(Big_Mart_Data.isnull().sum())

#=======================================================================================================================================================

#===============================
#2.Data Cleaning for Test
#===============================

# count the number of NaN values in each column
print(Big_Mart_Data_test.isnull().sum())

# Impute NA Values of Outlet_Size & 'Item_Weight

Big_Mart_impute(Big_Mart_Data_test)

# count the number of NaN values in each column after Imputation
print(Big_Mart_Data_test.isnull().sum())

#===============================
#3.Feature Engineering for Train
#===============================

#Big_Mart_Data columns transfering string to numeric
from sklearn.preprocessing import LabelEncoder
column_list=['Item_Identifier','Outlet_Size','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Location_Type','Outlet_Type']
for c in column_list:
    number=LabelEncoder()
    Big_Mart_Data[c]=number.fit_transform(Big_Mart_Data[c].astype('str'))
    Big_Mart_Data_test[c]=number.transform(Big_Mart_Data_test[c].astype('str'))
    
#===============================
#4.Modeling
#===============================

# split dataset into inputs and outputs
train_features = Big_Mart_Data.iloc[:,0:11]
train_labels = Big_Mart_Data.iloc[:,11]

from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_features, train_labels)


# Make predictions using the testing set
y_pred = regr.predict(Big_Mart_Data_test)


print('Predicted Values are:',y_pred)

