# -*- coding: utf-8 -*-
"""
Created on Thu Aug 06 17:01:35 2015

@author: deepsidhpura777
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import random

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

def xgbModel(X_train,y_train,X_test):
    
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.007
    params["min_child_weight"] = 8
    params["subsample"] = 0.5
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 10
   # params["colsample_bytree"] = 0.7
    plst = list(params.items())
    
    offset = 4000
    num_rounds = 10000
    

    xgtest = xgb.DMatrix(X_test)
    xgtrain = xgb.DMatrix(X_train[offset:,:],label=y_train[offset:])
    xgval = xgb.DMatrix(X_train[:offset,:],label=y_train[:offset])

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)
   # score_1 = model.get_fscore()

    X_train = X_train[::-1,:]
    y_train = np.log(y_train[::-1])

    xgtrain = xgb.DMatrix(X_train[offset:,:],label=y_train[offset:])
    xgval = xgb.DMatrix(X_train[:offset,:],label=y_train[:offset])

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)
   # score_2 = model.get_fscore()


    preds = preds1 * 3 + preds2 * 7

    return preds

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

    
    
def vector(train,test):
    
    X_train = train.ix[0:,1:]
    y_train = train.Hazard
    
    vec = DictVectorizer()
    X_train = X_train.T.to_dict().values()
    X_train = vec.fit_transform(X_train)
    
    X_test = test.T.to_dict().values()
    X_test = vec.fit_transform(X_test)
    
    return X_train,y_train,X_test
    
    

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submit = pd.read_csv('sample_submission.csv')

train = train.drop(['Id'],axis=1)
test = test.drop(['Id'],axis=1)

train_ = train.drop(['T2_V10','T2_V7','T1_V13','T1_V10'],axis = 1)
test_ = test.drop(['T2_V10','T2_V7','T1_V13','T1_V10'],axis = 1)

random.seed(1000)

originalList = original(train_,test_)
vectorList = vector(train,test)

predictionOriginal = xgbModel(originalList[0],originalList[1],originalList[2])
predictionVector = xgbModel(vectorList[0],vectorList[1],vectorList[2])

finalPrediction = (predictionOriginal ** 0.35) * 0.5 + (predictionVector ** 0.5) * 0.5  
submit.Hazard = finalPrediction

submit.to_csv('XGB_Last_Final.csv',index = False)


