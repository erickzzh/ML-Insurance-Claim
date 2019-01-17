#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:07:33 2019

@author: erickzhang
https://newonlinecourses.science.psu.edu/stat501/node/329/
"""
# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

#encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

columntransformer = ColumnTransformer([
    ("sex/smoker", OneHotEncoder(categories='auto'), [1,4]) 
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

'''-----Automatic Backward Elimination-----'''

import statsmodels.formula.api as sm
    
SLENTER,SLEXIT = 0.05,0.05
X = np.append(arr = np.ones((1338,1)).astype(int), values = X ,axis = 1 ) #axis = 0 add a line, axis = 1 add a coloumn
columnToKeep = []
columnToCheck = [i for i in range(0,len(X[0]))]
X = X.astype(float)
pEnterMin = 0
pLeaveMax = 1


while(pEnterMin<SLENTER or pLeaveMax>SLEXIT):
    forwardModels = {}
    backwardModels = {}

    #forward selection that selects 
    for i in columnToCheck:
        regressor_OLS = sm.OLS(endog = y,exog = X[ : ,columnToKeep+[i]]).fit()
        forwardModels[i]= regressor_OLS.pvalues[-1]
    
    # select the min p value from 
    columnWithSmallestP = min(forwardModels, key=forwardModels.get)
    pEnterMin = forwardModels[columnWithSmallestP]
    
    columnToKeep.append(columnWithSmallestP)
    columnToCheck.remove(columnWithSmallestP)
     
    #perform backward elimination
    regressor_OLS = sm.OLS(endog = y,exog = X[ : ,columnToKeep]).fit()
    
    for index,val in enumerate(columnToKeep):
        backwardModels[val]= regressor_OLS.pvalues[index]
        
    columnWithBiggestP = max(backwardModels, key=backwardModels.get)
    pLeaveMax = backwardModels[columnWithBiggestP]
    
    if(pLeaveMax > SLEXIT):
        columnToKeep.remove(columnWithBiggestP)
        columnToCheck.append(columnWithBiggestP) 
    
    
print( X[ : ,columnToKeep])  


