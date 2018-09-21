# -*- coding: utf-8 -*-
"""
Created on Sat Apr 7 10:01:56 2018

@author: mankayarkarasi.c
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, itertools, random 
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm, linear_model, model_selection
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def ReadCSVFile(file_path):
	data = pd.read_csv(file_path)
	return data 


def ReadExcelFile(file_path):
	data = pd.read_excel(file_path)
	return data 

def LabelEncode(column_list, t_dataset):
	for c in column_list:
		number=LabelEncoder()
		t_dataset[c]=number.fit_transform(t_dataset[c].astype('str'))
	return t_dataset 

def FillNAValueZero(column_list, t_dataset):
	for c in column_list:
		t_dataset[c].fillna(t_dataset[c].mode()[0],inplace=True)
	return t_dataset 

def PlotBarGraphdata(t_dataset,column_list):
	for c in column_list:
		plt.figure(1)
		plt.subplot(221)
		t_dataset[c].value_counts(normalize=True).plot.bar(figsize=(20,10), title= c)
		plt.show()
		
def PlotdataOutlier(t_dataset,column_list)	:
	for c in column_list:
		plt.figure(1)
		plt.subplot(121)
		sns.distplot(t_dataset[c]);
		plt.subplot(122)
		t_dataset[c].plot.box(figsize=(16,5))
		plt.show()	
		
def PlotCrossTab(t_dataset,column_list,traincolumn):
	for c in column_list:
		Xcolumn=pd.crosstab(t_dataset[c],t_dataset[traincolumn])
		Xcolumn.div(Xcolumn.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
		plt.show()
		
def PlobinGraph(t_dataset,bins,group,xaxis_col,yaxis_col, category, bincolumn):
	t_dataset[bincolumn]=pd.cut(t_dataset[xaxis_col],bins,labels=group)
	bincolumn=pd.crosstab(t_dataset[bincolumn],t_dataset[category])
	bincolumn.div(bincolumn.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
	plt.xlabel(xaxis_col)
	P = plt.ylabel(yaxis_col)
	
def dropcolumns(t_dataset,column_list):
	for c in column_list:
		t_dataset = t_dataset.drop(c, axis =1)
	return t_dataset

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(1)
    plt.xticks(tick_marks, rotation=45)
    plt.yticks(tick_marks)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plt.show()


def Kfold(test_data):
	i=1
	kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
	for train_index,test_index in kf.split(X,y):
			print('\n{} of kfold {}'.format(i,kf.n_splits))
			xtr, xvl =X.loc[train_index],X.loc[test_index]
			ytr,yvl = y[train_index],y[test_index]
			model = LogisticRegression(random_state=1)
			model.fit(xtr, ytr)
			pred_test = model.predict(xvl)
			score = accuracy_score(yvl,pred_test)
			print('accuracy_score',score)
			i+=1
			pred_test = model.predict(test_data)
			pred=model.predict_proba(xvl)[:,1]
	return X,y,xvl,yvl,pred
			
			
def PlotPositiveNegativeRate(vl,pred):
	
	fpr, tpr, _ = metrics.roc_curve(vl,  pred)
	auc = metrics.roc_auc_score(vl, pred)
	plt.figure(figsize=(12,8))
	plt.plot(fpr,tpr,label="validation, auc="+str(auc))
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc=4)
	plt.show()
			 
	
def AnalyseModelsLinearRadialDecisionTreeRandomForestLogisticRegression(X,y):
	# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators
	paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
	grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)
	x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)
	# Fit the grid search model
	grid_search.fit(x_train,y_train)

	# Estimating the optimized value
	grid_search.best_estimator_

	label = ['Linear SVM', 'Radial SVM', 'Decision Tree', 'Random Forest', 'Logistic Regression']
	models = [svm.LinearSVC(), svm.SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100), LogisticRegression()]
	accuracy = []
	fscore = []

	for model in models:
		   model.fit(x_train, y_train)
		   prediction = model.predict(x_cv)
		   accuracy.append(metrics.accuracy_score(prediction, y_cv))
		   fscore.append(metrics.f1_score(y_cv, prediction))
	model_df = pd.DataFrame({
							   'accuracy': accuracy,
							      'fscore': fscore
								  }, index=label)

	return model_df
	
	
def evaluate(model, x_train, y_train, x_test, y_test):
    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)
    train_acc = metrics.accuracy_score(y_train, train_preds)
    test_acc = metrics.accuracy_score(y_test, test_preds)
    print('Train accuracy: %s' % train_acc)
    print('Test accuracy: %s' % test_acc)

   
def split(data):
    # control randomization for reproducibility
    np.random.seed(42)
    random.seed(42)
    train, test = model_selection.train_test_split(data)
    x_train = train.loc[:, train.columns != 'Loan_Status']
    y_train = train['Loan_Status']
    x_test = test.loc[:, test.columns != 'Loan_Status']
    y_test = test['Loan_Status']
    return x_train, y_train, x_test, y_test

# ************Gradient Descent Optimization************
def split_train_evaluate(model, data):
    x_train, y_train, x_test, y_test = split(data)
    model.fit(x_train, y_train)
    evaluate(model, x_train, y_train, x_test, y_test)

