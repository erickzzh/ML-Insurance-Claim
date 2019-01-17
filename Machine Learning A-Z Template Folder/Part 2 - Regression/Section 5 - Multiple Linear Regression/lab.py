#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:23:58 2019

@author: erickzhang
"""

#Backward Elimination
'''
Step 1
Select a significance level to stay in the model SL usually equals to 0.05
Regress y on x1, y on x2, y on x3...

Step 2
Fit the full model with all possible predictors

Step 3
Consider the predictor with the highest P-val. If P>SL go to step 4 else you are done

P>SL means that the null hypothesis is confirment meaning that 
there is no significant difference or correlation between specified populations

Step 4
remove the predictor

Step 5 Fid model without this variable 
'''

#Forward Selection
'''
 Step 1
Select a significance level to stay in the model SL usually equals to 0.05

Step 2
Fit all simple regression model for instance if you have 100 independent variables 
you will have 100 models. Regress y on x1, y on x2, y on x3...

Step 3
Select the model with the lowest P value and add that variable to the other 99 models

Step 4
Find the lowest P value model and then add these predicotrs to the rest models

Step 5 
Repeat this process if P<SL otherwise go to fin and take the previous model.
'''

#Bidirectional Elimination
'''
Step 1 
Select a SL level to enter and to stay in the model
SLENTER = 0.05, SLSTAY= 0.05 

Step 2
Perform the next step of Forward Selection (new variables must have P<Slenter to enter)
Let's say we have 10 variables we will create 10 models and select ones with P < SLENTER

Step 3
Perform Backward Elimination 
Regress y on ONLY the variables that entered Step 3 and eliminate the one with the biggest
P value if it is bigger than SSTAY. Put that variable back to the pool and repeat Step 2

Step 4
No new variables can enter and no old varaibles can exit
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columntransformer = ColumnTransformer([
    ("State", OneHotEncoder(categories='auto'), [3]) 
], remainder='passthrough') 
 
X = columntransformer.fit_transform(X)

#Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results 
y_pred = regressor.predict(X_test)

#building backward elimination

'''
added a coloumn of 1s to fit the linear model formula

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis = 1 ) #axis = 0 add a line, axis = 1 add a coloumn

X_opt = X[:,[0,1,2,3,4,5]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:,[0,1,3,4,5]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary()) 

X_opt = X[:,[0,3,4,5]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary()) 

X_opt = X[:,[0,3,5]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary()) 

X_opt = X[:,[0,3]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary()) 


'''

'''-----Automatic Backward Elimination-----'''

import statsmodels.formula.api as sm
    
SL = 0.05
p_max = 1
X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis = 1 ) #axis = 0 add a line, axis = 1 add a coloumn
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(float)
while(p_max>SL):
    regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
    p_array = regressor_OLS.pvalues
    p_max = p_array.max()
    if(p_max > SL):
        ind_rem = np.where(p_array==p_max)[0][0]
        X_opt = np.delete(X_opt,ind_rem, 1)
        
print(X_opt)


'''forward selection'''
SL = 0.05
p_min = 0
#X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis = 1 ) #axis = 0 add a line, axis = 1 add a coloumn
columnToKeep = []
columnToCheck = [i for i in range(0,len(X[0]-1))]
X = X.astype(float)

regressor_OLS = sm.OLS(endog = y,exog = X).fit()
p_array = regressor_OLS.pvalues
p_min = p_array.min()

columnWithSmallestP = np.where(p_array==p_min)[0][0]
columnToKeep.append(columnWithSmallestP)
columnToCheck.remove(columnWithSmallestP)

while(p_min<SL):
    models = {}

    for i in range(0,len(columnToCheck)):
        regressor_OLS = sm.OLS(endog = y,exog = X[ : ,columnToKeep + [columnToCheck[i]]]).fit()
        models[columnToCheck[i]] = regressor_OLS.pvalues[-1]
        

    columnWithSmallestP = min(models, key=models.get)
    p_min = models[columnWithSmallestP]
    if(p_min <= SL):
        columnToKeep.append(columnWithSmallestP)
        columnToCheck.remove(columnWithSmallestP)
    else:
        pass 
print( X[ : ,columnToKeep])
  
  







