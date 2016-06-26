# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:10:18 2015

@author: deepsidhpura777
"""

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from hep_ml.losses import BinFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier
import numpy as np
import evaluation as e


import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

def knn(X,y,test):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, y)
    return neigh.predict_proba(test)[:,1]

def extra (X,y,test):
    
    clf = ExtraTreesClassifier(n_estimators=250, max_depth=9, min_samples_split=6)
    clf.fit(X,y)
    
    return clf.predict_proba(test)[:,1]



    
def NNet(X,y,test):
    
    
    
    #rows = int(X.shape[0])
    cols = int(X.shape[1])
    
    net = NeuralNet(
                   layers = [
                               ('input',layers.InputLayer),
                               ('hidden1',layers.DenseLayer),
                               ('dropout1',layers.DropoutLayer),
                               ('hidden2',layers.DenseLayer),
                               ('dropout2',layers.DropoutLayer),
                               ('hidden3',layers.DenseLayer),
                               ('dropout3',layers.DropoutLayer),
                               ('hidden4',layers.DenseLayer),
                               #('dropout4',layers.DropoutLayer),
                               ('output',layers.DenseLayer),
                            ],
                            input_shape = (None,cols),
                            hidden1_num_units = 70,
                            dropout1_p = 0.4,
                            hidden2_num_units = 50,
                            dropout2_p = 0.3,
                            hidden3_num_units = 30,
                            dropout3_p = 0.3,
                            hidden4_num_units = 15,
                            #dropout4_p = 0.2,
                            
                            
                            output_num_units = len(np.unique(y)),
                            output_nonlinearity = lasagne.nonlinearities.softmax,
    
                             update=nesterov_momentum,
                             update_learning_rate=0.01,
                             update_momentum=0.9,
                             max_epochs = 100,
                             verbose = 1,
                )
                
  #  net.load_params_from('w3')
  #  details = net.get_all_params()
  #  oldw = net.get_all_params_values()
    skf = cross_validation.StratifiedKFold(y,n_folds = 7)
    blend_train = np.zeros(X.shape[0])
    prediction = []
    blend_test_j = np.zeros((test.shape[0], len(skf)))
    
    for i,(train_index,cv_index) in enumerate(skf):
            print "Fold:",i
            X_train = X[train_index,]
            y_train = y[train_index]
            X_cv = X[cv_index,]
            #y_cv = y[cv_index]
            net.fit(X_train,y_train)
            
            blend_train[cv_index] = net.predict_proba(X_cv)[:,1]
            blend_test_j[:,i] = net.predict_proba(test)[:,1]
    prediction = blend_test_j.mean(1)
    
    return prediction
    


def flatnessloss(X,y,test):
    
    features = list(X.columns)
    features.remove('mass')
    loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
    clf = UGradientBoostingClassifier(loss=loss, n_estimators=300, subsample=0.7, 
                                  max_depth=9, min_samples_leaf=8,
                                  learning_rate=0.1, train_features=features, random_state=11)

    
    
    arr = np.random.permutation(X.shape[0])    
    X = X.ix[arr,]
    y = y[arr]
    
    
    skf = cross_validation.StratifiedKFold(y,n_folds = 7)
    blend_train = np.zeros(X.shape[0])
    prediction = []
    blend_test_j = np.zeros((test.shape[0], len(skf)))
    
    for i,(train_index,cv_index) in enumerate(skf):
            print "Fold:",i
            X_train = X.ix[train_index,]
            y_train = y[train_index]
            X_cv = X.ix[cv_index,]
            #y_cv = y[cv_index]
            clf.fit(X_train,y_train)
            
            blend_train[cv_index] = clf.predict_proba(X_cv)[:,1]
            blend_test_j[:,i] = clf.predict_proba(test)[:,1]
    prediction = blend_test_j.mean(1)
        
    return prediction
    



def rfModel(X,y,test):
    
    skf = cross_validation.StratifiedKFold(y,n_folds = 7)
    clfs = [RandomForestClassifier(n_estimators=300,criterion = "entropy"),
            RandomForestClassifier(n_estimators=150,criterion = "entropy") ]
            
    blend_train = np.zeros((X.shape[0], len(clfs)))
    blend_test = np.zeros((test.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        
        print 'Training classifier [%s]' % (j)
        blend_test_j = np.zeros((test.shape[0], len(skf)))
        
        for i,(train_index,cv_index) in enumerate(skf):
            print "Fold:",i
            X_train = X[train_index]
            y_train = y[train_index]
            X_cv = X[cv_index]
            #y_cv = y[cv_index]
            clf.fit(X_train,y_train)
            
            blend_train[cv_index, j] = clf.predict_proba(X_cv)[:,1]
            blend_test_j[:,i] = clf.predict_proba(test)[:,1]
        blend_test[:,j] = blend_test_j.mean(1)
            
    #feature_imp = m.feature_importances_
    prediction = (blend_test[:,0] + blend_test[:,1]) / 2
    return prediction

def xgbModel(X,y,test):
    
    
    
    params = {}
    params["objective"] = "binary:logistic"
    params["eta"] = 0.1
    params["min_child_weight"] = 10
    params["subsample"] = 0.7
    params["scale_pos_weight"] = 0.8
    params["silent"] = 1
    params["max_depth"] = 7
    
    plst = list(params.items())
    
    xgtrain = xgb.DMatrix(X,label = y)
    xgtest = xgb.DMatrix(test)
    
    num_rounds = 1000
    
    model = xgb.train(plst, xgtrain, num_rounds)
    prediction = model.predict(xgtest)
        
    
    return prediction
  
  
  
def check_a(agreement_probs):
    check_agreement = pd.read_csv('check_agreement.csv')
    ks = e.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print 'KS metric', ks, ks < 0.09
    
    

train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')

np.random.seed(671)


# read test values
# drop some columns not there in test data set.......

y = train['signal']
XwithMass = train.drop(['id','production','min_ANNmuon','signal','SPDhits'],axis = 1)
X = train.drop(['id','mass','production','min_ANNmuon','signal','SPDhits'],axis = 1)
test = test.drop(['id','SPDhits'],axis = 1)


#p4 = flatnessloss(XwithMass,y,test)

X = X.values
y = y.values
test = test.values


X = X.astype('float32')
y = y.astype('int32')
test = test.astype('float32')


arr = np.random.permutation(X.shape[0])    
X = X[arr,]
y = y[arr]
  
#p1 = rfModel(X,y,test)



arr = np.random.permutation(X.shape[0])    
X = X[arr,]
y = y[arr]

p2 = xgbModel(X,y,test) 

## Neural Networks starts

arr = np.random.permutation(X.shape[0])    
X = X[arr,]
y = y[arr]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
test = scaler.fit_transform(test)

#p3 = NNet(X,y,test)


sub.prediction = (p1 + p2 + p3 + p4) / 4 
check_a(sub.prediction)
sub.to_csv('submission.csv',index = False)
