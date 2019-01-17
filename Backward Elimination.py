#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:28:20 2019

@author: ErickZhang
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    
SL = 0.05
p_max = 1
X = np.append(arr = np.ones((1338,1)).astype(int), values = X ,axis = 1 ) #axis = 0 add a line, axis = 1 add a coloumn
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
