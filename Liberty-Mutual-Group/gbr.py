# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 10:53:25 2015

@author: deepsidhpura777
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def gbrModel(X_train,y_train,X_test) :
    
    gbr = GradientBoostingRegressor(n_estimators=100,max_depth=5)
    model = gbr.fit(X_train,y_train)
    prediction = model.predict(X_test)
    
    return prediction


def convert(X) :
    for i in range(X.shape[1]):
        dict = {}
        names = []
        col = X.ix[:,i]
        if col.dtype == 'O' :
                names = list(enumerate(np.unique(col)))
                dict = {name : j for j, name in names}
                X.ix[:,i] = X.ix[:,i].map(lambda x : dict[x]).astype(int)
    return X
    
def original(train,test):
    
    X_train = train.ix[0:,1:]
    y_train = train.Hazard

    X_train = convert(X_train)
    X_test = convert(test[:])

    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train,y_train,X_test
    
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submit = pd.read_csv('sample_submission.csv')

train = train.drop(['Id'],axis=1)
test = test.drop(['Id'],axis=1)

train_ = train.drop(['T2_V10','T2_V7','T1_V13','T1_V10'],axis = 1)
test_ = test.drop(['T2_V10','T2_V7','T1_V13','T1_V10'],axis = 1)


originalList = original(train_,test_)
predictionGBR = gbrModel(originalList[0],originalList[1],originalList[2])

submit.Hazard = predictionGBR

