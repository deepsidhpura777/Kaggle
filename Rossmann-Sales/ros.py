# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:57:22 2015

@author: deepsidhpura777
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def gbr(X,y,test):
    
    y = np.log1p(y)    
    clf = GradientBoostingRegressor(n_estimators = 400)
    clf.fit(X,y)
    prediction = np.expm1(clf.predict(test))
    
    return prediction

def xgbModel(X,y,test):
    
    params = {}
    
    params["objective"] = "reg:linear"
    params["booster"] = "gbtree"
    params["eta"] = 0.15
    params["min_child_weight"] = 7
    params["subsample"] = 0.8
    params["scale_pos_weight"] = 0.8
    params["silent"] = 1
    params["max_depth"] = 10
    #params["max_delta_step"]=2

    
    plst = list(params.items())
    
    offset = 10000
    
    num_rounds = 3500
    
    y = np.log1p(y)
    xgtest = xgb.DMatrix(test)
        
    xgtrain = xgb.DMatrix(X[offset:,:],label=y[offset:])
    xgval = xgb.DMatrix(X[:offset,:],label=y[:offset])
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    
    model = xgb.train(plst, xgtrain, num_rounds,evals=watchlist,early_stopping_rounds=100,feval=rmspe_xg,verbose_eval=True)
    prediction = model.predict(xgtest)
    prediction = np.expm1(prediction)
    
    return prediction
    


    
train = pd.read_csv('newTrain.csv')
y = train['Sales']
X = train.drop(['Sales','Customers'],axis = 1)
test = pd.read_csv('newTest.csv')
Id = test['Id']
test = test.drop(['Id'],axis = 1)


X = X.values
y = y.values
test = test.values

gbrPrediction = gbr(X,y,test)



