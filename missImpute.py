# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:40:12 2017

"""
import numpy as np
import itertools
from copy import deepcopy as copy

''' GenX, GenY, and GenMiss are only used in simulation to compare the performance of different way of missing value imputations.
    hence, not needed in usual usage case of MissingImputation2 '''

def GenX(n,d):
    return np.random.randn(n,d)
    
class GenY(object):
    # simulate y as multilayer perceptron of X
    def __init__(self,dim,nonFun,bias=0,noise=1e-1):
        self.dim = dim
        self.nonFun = nonFun
        self.noise = noise
        self.bias = bias
        self.beta = []
        for i in range(len(dim)-1):
            self.beta.append(np.random.randn(dim[i],dim[i+1])/2)
    
    def predict(self,X):
        for para in self.beta:
            X = self.nonFun(np.dot(X,para))
        return np.argmax(X + self.bias + np.random.randn(*X.shape) * self.noise,1)
        
class GenMiss(object):
    # simulate a masked version of X, where masked X is missing and later filled in by imputation  
    def __init__(self,betaGen,d,nonFun=None,bias=0,noise=1e-1):
        # betaGen == np.random.randn is the most general case
        # whereas == np.zeros assumes missing at random
        self.beta = betaGen(d+1,d)
        self.noise = noise
        self.bias = bias
        self.nonFun = nonFun
    
    def predict(self,X,y):
        X_miss = copy(X)
        if self.nonFun is None:
            X_miss[np.dot(np.c_[X,y],self.beta) + np.random.randn(*X.shape) * self.noise > self.bias] = np.nan
        else:                
            X_miss[self.nonFun(np.dot(np.c_[X,y],self.beta)) + np.random.randn(*X.shape) * self.noise > self.bias] = np.nan
        return X_miss
        
def allComb2(VecIn_,IndexIn_,RemList_,X,lower,upper):
    
    if len(RemList_) == 0:
        temp = np.sum(VecIn_)
        if temp<upper:
            return [VecIn_],[IndexIn_]
        else:
            return [],[]
    else:
        curr = RemList_.pop()
        leftVec,leftIndex = allComb2(VecIn_,copy(IndexIn_),copy(RemList_),X,lower,upper) # does not have curr
        
        temp2 = VecIn_*X[:,curr]
        if np.sum(temp2)>lower: # prune tree
            IndexIn_.append(curr)
            rightVec,rightIndex = allComb2(temp2,IndexIn_,RemList_,X,lower,upper) # have curr
            leftVec.extend(rightVec)
            leftIndex.extend(rightIndex)

        return leftVec, leftIndex
        
def combArray(X1,X2):
    return np.stack([X1[:,i]*X2[:,j] for i,j in itertools.product(range(X1.shape[1]),range(X2.shape[1]))],1)
    
class MissingImputation2(object):
    # Create missing dummies for all variables and all possible interaction(2^d of them) 
    # terms of dummies created that satisfies the sparsity bounds. 
    def __init__(self,lower,upper):
        self.lower = lower
        self.upper = upper
        self.interactionList = []
    
    def fit(self,X): 
        X = copy(X)
        n,d = X.shape
        self.mean = np.nanmean(X)
        lower, upper = n * self.lower, n * self.upper
        X_nan = np.isnan(X)
        X_transf,self.interactionList = allComb2(np.ones(n,dtype='bool'),[],range(d),X_nan,lower,upper)
        X_transf = np.stack(X_transf,1)
        X[X_nan] = self.mean
        return np.c_[X,X_transf,combArray(X,X_transf)]
    
    def predict(self,X):
        X = copy(X)
        n,_ = X.shape
        d = len(self.interactionList)
        X_transf = np.ones((n,d),dtype='bool')
        X_nan = np.isnan(X)
        X[X_nan] = self.mean
        for count_,i in enumerate(self.interactionList):
            for j in i:
                X_transf[:,count_] = X_transf[:,count_] * X_nan[:,j]
        return np.c_[X,X_transf,combArray(X,X_transf)]
        
class SimplyImputation(object):
    # fill in marginal mean/median
    def __init__(self,method='mean'):
        self.method = method
    
    def fit(self,X): 
        self.para = np.nanmean(X,0) if self.method == 'mean' else np.nanmedian(X,0)
        X_transf = np.ones_like(X) * self.para
        index_ = np.logical_not(np.isnan(X))
        X_transf[index_] = X[index_]
        return X_transf
    
    def predict(self,X):
        X_transf = np.ones_like(X) * self.para
        index_ = np.logical_not(np.isnan(X))
        X_transf[index_] = X[index_]
        return X_transf    
        
