# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:12:26 2018

@author: mankayarkarasi.c
"""

import os
os.chdir('C:\\Users\\mankayarkarasi.c\\Desktop\\AIML\\Regression')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


#===============================
#Data Exploration
#===============================

blackFriday_df = pd.read_csv('TrainNew.csv')

blackFriday_df.apply(lambda x: len(x.unique()))
#By looking at the dataset it is observed that all the columns except
#User ID, Purchase ID and Purchase amount are categorical data and not 
#any distribution
tt = blackFriday_df.head()

blackFriday_df.info()
#Nulls Present in only the columns - Product Category 2 & 3

#===============================
#Data Cleaning for Train
#===============================

#Since it is categorical, filling nulls with 0
blackFriday_df.fillna(0, inplace = True)

blackFriday_df = blackFriday_df.drop_duplicates()

blackFriday_df.drop(['Name', 'City_Name'], inplace = True , axis = 1)
#Demographic Variables


sns.set()
cols = ['Purchase', 'Gender', 'Age', 'Occupation' , 'City_Category',
        'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
        'Product_Category_2', 'Product_Category_3']
sns.pairplot(blackFriday_df[cols])
plt.show()
#Wanted to check if any variable is similarly distributed as Purchase
#Observed that all are categorical variables

sns.barplot(x = blackFriday_df['Age'], y = blackFriday_df['Purchase'])
#Almost equally distributed

sns.barplot(x = blackFriday_df['Gender'], y = blackFriday_df['Purchase'])
#Almost equally distributed

ax = sns.regplot(x="Age", y="City_Category", data=blackFriday_df)
#Age and City Category as equally disctributed


sns.distplot(blackFriday_df['Purchase'])
print("Skewness: %f" % blackFriday_df['Purchase'].skew())

sns.distplot(blackFriday_df['Purchase'][blackFriday_df['City_Category'] == 'A'])

sns.distplot(blackFriday_df['Purchase'][blackFriday_df['City_Category'] == 'B'])

sns.distplot(blackFriday_df['Purchase'][blackFriday_df['City_Category'] == 'C'])

sns.distplot(blackFriday_df['Purchase'][blackFriday_df['Gender'] == 'M'])

sns.distplot(blackFriday_df['Purchase'][blackFriday_df['Gender'] == 'F'])
#From above 5 graphs, we see that the purchase is equally distributed among cities and genders

user_id = blackFriday_df[['User_ID','Product_ID']]
prods = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']
columns = ['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status']

X = blackFriday_df[['User_ID','Product_ID', 'Purchase'] + columns + prods]


from sklearn.preprocessing import LabelEncoder

for c in columns + prods:
    labelencoder_X = LabelEncoder()
    X[c] = labelencoder_X.fit_transform(X[c])

def correlation_heatmap(df, title):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(10, 240, as_cmap = True)
    
    corr_matrix = df.corr()
    upper = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(np.bool))
    
    _ = sns.heatmap(
        upper,
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True,
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title(title, y=1.05, size=15)

correlation_heatmap(X, 'Correlation')


#X2 = onehotencoder_X.fit_transform(X[columns + prods]).toarray()
X21 = pd.get_dummies(X, columns = columns + prods)

dummy_trap = [a+'_0' for a in columns + prods]

y = X21['Purchase']

X21 = X21.drop(dummy_trap + ['Purchase'], axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X21, y, test_size = 0.3)

#===============================
#Model Building
#===============================

from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train.iloc[:, 2:], y_train.values)


# Make predictions using the testing set
y_pred = regr.predict(X_test.iloc[:, 2:])

y_test2 = y_test.values
from math import sqrt
from sklearn.metrics import mean_squared_error

sqrt(mean_squared_error(y_test2, y_pred))


SS_Residual = sum((y_test2-y_pred)**2)
SS_Total = sum((y_test2-np.mean(y_test2))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test2)-1)/(len(y_test2)-X_train.iloc[:, 2:].shape[1]-1)
print (r_squared, adjusted_r_squared)


train = pd.read_csv('TrainNew.csv')
test = pd.read_csv('Predict.csv')

y = train['Purchase']

train.drop(['Purchase'], inplace = True, axis = 1)

train['Type'] = 'Train'
test['Type'] = 'Test'

df = pd.concat([train, test])

df.drop(['Name', 'City_Name'], inplace = True , axis = 1)

#Since it is categorical, filling nulls with 0
df.fillna(0, inplace = True)

df = df.drop_duplicates()

from sklearn.preprocessing import LabelEncoder

for c in columns + prods :
    labelencoder_X = LabelEncoder()
    df[c] = labelencoder_X.fit_transform(df[c])
   

df = pd.get_dummies(df, columns = columns + prods)

df = df.drop(dummy_trap, axis = 1)


X_train = df[df['Type'] == 'Train']
X_test = df[df['Type'] == 'Test']

X_train.drop(['Type'], inplace = True, axis = 1)
X_test.drop(['Type'], inplace = True, axis = 1)

from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(X_train.iloc[:, 2:], y.values)

y_pred = regr.predict(X_test.iloc[:, 2:])
