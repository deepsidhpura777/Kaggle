# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:41:12 2015

@author: deepsidhpura777
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing


def convert(X) :
    for i in range(X.shape[1]):
        dic = {}
        names = []
        col = X.ix[:,i]
        if col.dtype == 'O' :
                names = list(enumerate(np.unique(col)))
                dic = {name : j for j, name in names}
                X.ix[:,i] = X.ix[:,i].map(lambda x : dic[x]).astype(int)
    return X
    
def data_processing(train,test,tube,bill):
    
   train = pd.merge(train, tube, on ='tube_assembly_id')
   train = pd.merge(train, bill, on ='tube_assembly_id')
   test = pd.merge(test, tube, on ='tube_assembly_id')
   test = pd.merge(test, bill, on ='tube_assembly_id')



# create some new features
   train['year'] = train.quote_date.dt.year
   train['month'] = train.quote_date.dt.month
#train['dayofyear'] = train.quote_date.dt.dayofyear
#train['dayofweek'] = train.quote_date.dt.dayofweek
#train['day'] = train.quote_date.dt.day

   test['year'] = test.quote_date.dt.year
   test['month'] = test.quote_date.dt.month
#test['dayofyear'] = test.quote_date.dt.dayofyear
#test['dayofweek'] = test.quote_date.dt.dayofweek
#test['day'] = test.quote_date.dt.day

# drop useless columns and create labels
#idx = test.id.values.astype(int)
   test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
   y_train = train.cost.values
#'tube_assembly_id', 'supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x'
#for some reason material_id cannot be converted to categorical variable
   train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

   train['material_id'].replace(np.nan,' ', regex=True, inplace= True)
   test['material_id'].replace(np.nan,' ', regex=True, inplace= True)
   for i in range(1,9):
       column_label = 'component_id_'+str(i)
       print(column_label)
       train[column_label].replace(np.nan,' ', regex=True, inplace= True)
       test[column_label].replace(np.nan,' ', regex=True, inplace= True)

   train.fillna(0, inplace = True)
   test.fillna(0, inplace = True)



# convert data to numpy array
   train = np.array(train)
   test = np.array(test)


# label encode the categorical variables
   for i in range(train.shape[1]):
        if i in [0,3,5,11,12,13,14,15,16,20,22,24,26,28,30,32,34]:
            print(i,list(train[1:5,i]) + list(test[1:5,i]))
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[:,i]) + list(test[:,i]))
            train[:,i] = lbl.transform(train[:,i])
            test[:,i] = lbl.transform(test[:,i])


# object array to float
   X_train = train.astype(float)
   X_test = test.astype(float)
    
   return X_train,y_train,X_test
    
    
def xgbModel(X_train,y_train,X_test):
    
    params = {}
    
    params["objective"] = "reg:linear"
    params["eta"] = 0.03
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["scale_pos_weight"] = 0.8
    params["silent"] = 1
    params["max_depth"] = 9
    params["max_delta_step"]=2

    
    plst = list(params.items())
    
    offset = 4000
    #roundsList = [2000,3000,4000,4000]
    #mini = 1
   # prediction = []
    xgtest = xgb.DMatrix(X_test)
    y_train = np.power(y_train,1/16.0)
    
    for i in range(1):
        
       # num_rounds = roundsList[i]  
        num_rounds = 5000
        
        xgtrain = xgb.DMatrix(X_train[offset:,:],label=y_train[offset:])
        xgval = xgb.DMatrix(X_train[:offset,:],label=y_train[:offset])
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    
        m1 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=150)
        p1 = m1.predict(xgtest,ntree_limit=m1.best_iteration)
        p1 = np.power(p1,16.0)
       # cv1 = m1.best_score
    
    
    
        X_train = X_train[::-1,:]
        y_train = y_train[::-1]
        xgtrain = xgb.DMatrix(X_train[offset:,:],label=y_train[offset:])
        xgval = xgb.DMatrix(X_train[:offset,:],label=y_train[:offset])
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    
        m2 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=150)
        p2 = m2.predict(xgtest,ntree_limit=m2.best_iteration)
        p2 = np.power(p2,16.0) ## Root transformation to make the output more normalized.
        
        
        
        
        arr = np.random.permutation(X_train.shape[0])    
        X_train = X_train[arr]
        y_train = y_train[arr]
    
        xgtrain = xgb.DMatrix(X_train[offset:,:],label=y_train[offset:])
        xgval = xgb.DMatrix(X_train[:offset,:],label=y_train[:offset])
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    
        m3 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=150)
        p3 = m1.predict(xgtest,ntree_limit=m3.best_iteration)
        p3 = np.power(p3,16.0)
       # cv1 = m1.best_score
    
    
    
        X_train = X_train[::-1,:]
        y_train = y_train[::-1]
        xgtrain = xgb.DMatrix(X_train[offset:,:],label=y_train[offset:])
        xgval = xgb.DMatrix(X_train[:offset,:],label=y_train[:offset])
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    
        m4 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=150)
        p4 = m2.predict(xgtest,ntree_limit=m4.best_iteration)
        p4 = np.power(p4,16.0) ## Root transformation to make the output more normalized.
        #cv2 = m2.best_score
       # cv = (cv1 + cv2) * 0.5
       #if cv < mini:
           # mini = cv
        p = (p1 + p2 + p3 + p4) * 0.25
           
    
    return p

def xgbFull(X_train,y_train,X_test):
    
    params = {}
    
    params["objective"] = "reg:linear"
    params["eta"] = 0.02
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["scale_pos_weight"] = 0.8
    params["silent"] = 1
    params["max_depth"] = 8
    params["max_delta_step"]=2
    plst = list(params.items())
    
    
    xgtest = xgb.DMatrix(X_test)
    y1 = np.log1p(y_train)
    y2 = np.power(y_train,1/16.0)
    
    num_rounds = 1000
    print(num_rounds)
    xgtrain = xgb.DMatrix(X_train,label=y1)
    m1 = xgb.train(plst,xgtrain,num_rounds)
    p1 = m1.predict(xgtest)
    p1 = np.expm1(p1)
    
    num_rounds = 2000
    print(num_rounds)
  
    xgtrain = xgb.DMatrix(X_train,label=y2)
    m2 = xgb.train(plst,xgtrain,num_rounds)
    p2 = m2.predict(xgtest)
    p2 = np.power(p2,16.0)
    
    num_rounds = 3000
    print(num_rounds)
    xgtrain = xgb.DMatrix(X_train,label=y1)
    m3 = xgb.train(plst,xgtrain,num_rounds)
    p3 = m3.predict(xgtest)
    p3 = np.expm1(p3)
    
    num_rounds = 4000
    print(num_rounds)
    xgtrain = xgb.DMatrix(X_train,label=y2)
    m4 = xgb.train(plst,xgtrain,num_rounds)
    p4 = m4.predict(xgtest)
    p4 = np.power(p4,16.0)
    
    
   
        
    return p1,p2,p3,p4
    
    

train = pd.read_csv('train_set.csv',parse_dates = [2,])
test = pd.read_csv('test_set.csv',parse_dates = [3,] )
sub = pd.read_csv('sample_submission.csv')
tube = pd.read_csv('tube.csv')
bill = pd.read_csv('bill_of_materials.csv')


dataList = data_processing(train,test,tube,bill)

a = xgbFull(dataList[0],dataList[1],dataList[2])
#a2 = xgbModel(dataList[0],dataList[1],dataList[2])

s =  a[0] * 0.30 + a[1] * 0.20 + a[2] * 0.20 + a[3] * 0.30

sub.cost = s
sub.to_csv('XGB_NEW_Hopeful.csv',index = False)
