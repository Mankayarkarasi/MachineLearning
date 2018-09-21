# -*- coding: utf-8 -*-
"""
Created on Sat Apr 7 11:13:32 2018

@author: mankayarkarasi.c
"""

from DataProcessTemplate import *


train_fiepath = "C:\\Users\\mankayarkarasi.c\\Desktop\\AIML\\Classification\\Training Set.csv"

predict_fiepath = "C:\\Users\\mankayarkarasi.c\\Desktop\\AIML\\Classification\\Prediction_Results.csv"

data = ReadCSVFile(train_fiepath)
train_data = ReadCSVFile(train_fiepath)
test_data = ReadCSVFile(train_fiepath)
train_original=train_data.copy()
test_original=ReadCSVFile(predict_fiepath)


data.head()
data.describe()
data.describe(include=['O'])
data.shape
data.isnull().sum()

train_data.columns
train_data.dtypes
 

train_data.isnull().sum()
column_list = ['Gender','Married','Dependents','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']
FillNAValueZero(column_list, train_data)

data.isnull().sum()
column_list = ['Gender','Married','Dependents','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']
FillNAValueZero(column_list, data)

test_data.isnull().sum()
column_list = ['Gender','Married','Dependents','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']
FillNAValueZero(column_list, test_data)

train_data.describe()
train_data.describe(include=['O'])
train_data.shape

# Hypothesis:

#Chances of loan approval are better if Credit History is good
#Chances of loan approval are better if Education is Graduate or higher
#Chances of loan approval are better if Applicant income is higher
#Chances of loan approval are better if Co-pplicant income is higher
#Chances of loan approval are better if tenure of loan is Less
#Chances of loan approval are better if the loan amount is less
#Chances of loan approval are better if number of dependents is less

column_list = ['Gender','Married','Dependents','Self_Employed','Education','Property_Area','Credit_History','Loan_Status']
PlotBarGraphdata(train_data,column_list)

#Conclusion:

#80% applicants are male.
#Around 65% applicants are married.
#Around 15% applicants iare self employed.
#Around 85% applicants have paid their debt.

# Univariate Analysis - Independent Variables - Ordinal

# Conclusion
# Most of the applicants don’t have any dependents.
# Around 80% of the applicants are Graduate.
# Most of the applicants are from Semiurban area.

# Univariate Analysis - Independent Variables - Numerical

column_list = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
PlotdataOutlier(train_data,column_list)


# Conclusion: distribution of applicant income is right skewed and not normally distributed.
# The boxplot confirms the presence of a lot of outliers/extreme values, due to income disparity
#Outlier Treatment for continuous features


train_data['LoanAmount_log'] = np.log(train_data['LoanAmount'])
train_data['LoanAmount_log'].hist(bins=20)
test_data['LoanAmount_log'] = np.log(test_data['LoanAmount'])

#segregate them by Education

train_data.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")

# Bi-variate Analysis
# Categorical Independent Variable vs Target Variable

column_list = ['Gender','Married','Dependents','Self_Employed','Education']
PlotCrossTab(train_data,column_list,'Loan_Status')

# Conclusion: No major impact of Gender bias in loan processing
# married applicants approval rate is higher.
# Numerical Independent Variable vs Target Variable

train_data.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

# We don’t see any change in the mean income, so lets take bins.

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
PlobinGraph(train_data,bins,group,'ApplicantIncome','Percentage','Loan_Status','Income_bin')

column_list = ['Credit_History','Property_Area']
PlotCrossTab(train_data,column_list,'Loan_Status')



bins=[0,1000,3000,42000]
group=['Low','Average','High']
PlobinGraph(train_data,bins,group, 'CoapplicantIncome','Percentage','Loan_Status','Coapplicant_Income_bin')

# Graph shows High co-applicant income has higher rejection rate, which should idealy not be the case
# Thus ,try Combining Applicant Income and Coapplicant Income and evaluate combined effect of Total Income on loan approval.

train_data['Total_Income'] = train_data['ApplicantIncome'] + train_data['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
PlobinGraph(train_data,bins,group, 'Total_Income','Percentage','Loan_Status','Total_Income_bin')

#This gives much better influence of Total (Applicant +co applicant income on the approval rate
# As the Total income grows approval rate improves

test_data['Total_Income']=test_data['ApplicantIncome']+test_data['CoapplicantIncome']

column_list = ['Total_Income']
PlotdataOutlier(test_data,column_list) 

train_data['Total_Income_log'] = np.log(train_data['Total_Income'])
sns.distplot(train_data['Total_Income_log']);
test_data['Total_Income_log'] = np.log(test_data['Total_Income'])

# Evaluate impact of Next Numeric Variable - Loan Amount on the approval rate

bins=[0,100,200,700]
group=['Low','Average','High']
PlobinGraph(train_data,bins,group, 'LoanAmount','Percentage','Loan_Status','LoanAmount_bin')

#Conclusion : Percentage of approved loans is higher for Low and Average Loan Amount as compared High Loan Amount 
#This supports our hypothesis
#Thus chances of loan approval will be high when the loan amount is less.
# drop the bins which we created for the exploration
train_data=train_data.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin'], axis=1)

train_data=train_data.drop(['LoanAmount_log','Total_Income','Total_Income_log'], axis=1)


# Feature Engineering
# Normalize Features
#Convert to Numeric Values for processing model
train_data['Dependents'].replace('3+', 3,inplace=True)
test_data['Dependents'].replace('3+', 3,inplace=True)
data['Dependents'].replace('3+', 0,inplace=True)
train_data['Loan_Status'].replace('N', 0,inplace=True)
train_data['Loan_Status'].replace('Y', 1,inplace=True)
data['Loan_Status'].replace('Y', 1,inplace=True)

# Correlation Map -correlation between all the numerical variables

plt.figure(figsize=(15, 7))
sns.heatmap(train_data.corr(), annot=True)

test_data.columns

train_data.columns

columns = ['Loan_ID','CoapplicantIncome','ApplicantIncome']
train_data= dropcolumns(train_data,columns)
test_data = dropcolumns(test_data,columns)



# ********************   Model Building 	******************** #


X = train_data.drop('Loan_Status',1)
y = train_data.Loan_Status 
X=pd.get_dummies(X)
train_data=pd.get_dummies(train_data)
test_data=pd.get_dummies(test_data)

x_train,x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.3)

model = LogisticRegression()
model.fit(x_train,y_train)
pred_cv = model.predict(x_cv)

# Evaluating Model Accuracy
accuracy_score(y_cv, pred_cv)

# Model Accuracy is better than 82 %
print(pred_cv)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_cv, pred_cv)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True,
                      title='Normalized confusion matrix')



test_data.columns
columns = ['Loan_Status_N','Loan_Status_Y']
test_data= dropcolumns(test_data,columns)

# Make Predictions on Test Data
pred_test = model.predict(test_data)

x_cv.head()

xvl,yvl, pred = Kfold(X,y,test_data)

fpr, tpr, _ = metrics.roc_curve(yvl,  pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


#Create New Variable "EMI" 
#Hypothesis is if EMI is less approval rate is better

train_data['EMI']=train_data['LoanAmount']/train_data['Loan_Amount_Term']
test_data['EMI']=test_data['LoanAmount']/test_data['Loan_Amount_Term']

train_data['EMI_log'] = np.log(train_data['EMI'])
test_data['EMI_log'] = np.log(test_data['EMI'])

#Create New Variable "Balance income" 
#Hypothesis is if Balance income is high approval rate is better

train_data['Balance Income']=train_data['Total_Income']-(train_data['EMI']*1000) # Multiply with 1000 to make the units equal 
test_data['Balance Income']=test_data['Total_Income']-(test_data['EMI']*1000)

train_data['Balance Income_log'] = np.log(train_data['Balance Income'])
test_data['Balance Income_log'] = np.log(test_data['Balance Income'])
test_data['Balance Income_log'].fillna(test_data['Balance Income_log'].median(), inplace=True)


sns.distplot(train_data['EMI'])
sns.distplot(train_data['EMI_log'])
sns.distplot(train_data['Balance Income'])

sns.distplot(train_data['Balance Income_log'])

train_data=train_data.drop(['LoanAmount', 'Loan_Amount_Term'], axis=1)
test_data=test_data.drop(['LoanAmount', 'Loan_Amount_Term'], axis=1)

train_data.columns

train_data=train_data.drop(['Balance Income', 'EMI'], axis=1)
test_data=test_data.drop(['Balance Income', 'EMI'], axis=1)

test_data.isnull().sum()

X,y,xvl,yvl,pred = Kfold(test_data)

AnalyseModelsLinearRadialDecisionTreeRandomForestLogisticRegression(X,y)

train_data.isnull().sum()
#
train_data['Balance Income_log'] = train_data['Balance Income_log'].round(4)
test_data['Balance Income_log'] = test_data['Balance Income_log'].round(4)


train_data['EMI_log'] = train_data['EMI_log'].round(4)
test_data['EMI_log'] = test_data['EMI_log'].round(4)


train_data['LoanAmount_log'] = train_data['LoanAmount_log'].round(4)
test_data['LoanAmount_log'] = test_data['LoanAmount_log'].round(4)

train_data['Total_Income_log'] = train_data['Total_Income_log'].round(4)
test_data['Total_Income_log'] = test_data['Total_Income_log'].round(4)


train_data['Total_Income'] = train_data['Total_Income'].round(4)
test_data['Total_Income'] = test_data['Total_Income'].round(4)

# ***************** Create Output File **********************#
output=train_original
output['Loan_ID']=train_original['Loan_ID']
output.head()

pd.DataFrame(output, columns=['Loan_ID','Loan_Status']).to_csv("C:\\Users\\mankayarkarasi.c\\Desktop\\AIML\\Classification\\Submission.csv")
output.head()

train_data['Balance Income_log'].fillna(train_data['Balance Income_log'].median(), inplace=True)

x_train, y_train, x_test, y_test = split(train_data)
grid_search = model_selection.GridSearchCV(
    estimator=linear_model.SGDClassifier(loss='log'),
    param_grid={'alpha': [0.01, 0.1, 1.],
                'max_iter': [1000, 10000]},
    cv=10,
    return_train_score=True)
grid_search.fit(x_train, y_train)


r = pd.DataFrame(grid_search.cv_results_)

best_model = grid_search.best_estimator_
print(grid_search.best_params_)
evaluate(best_model, x_train, y_train, x_test, y_test)





