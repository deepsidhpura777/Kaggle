# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 23:18:47 2015

@author: deepsidhpura777
"""
import numpy as np

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
    

    




            