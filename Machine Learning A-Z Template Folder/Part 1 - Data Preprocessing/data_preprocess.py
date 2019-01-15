#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:25:43 2019

@author: erickzhang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
data = pd.read_csv('Data.csv')
X = pd.DataFrame(data.iloc[:, :-1].values)
y = pd.DataFrame(data.iloc[:,3].values)

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:,1:3]) #selects the two coloumns that have NaN 
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])



#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0]) #convert country names into number coding

'''
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray() #convert country names into number coding
'''

#or we can do this way
columntransformer = ColumnTransformer([
    ("Countries", OneHotEncoder(categories='auto'), [0]) 
], remainder='passthrough') 
 
X = columntransformer.fit_transform(X)
 
#convert YES/NO names into number coding
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)




#now we need to split the traning data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)


#now we need to normalize all the values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)