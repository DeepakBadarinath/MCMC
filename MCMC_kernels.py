# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 07:28:48 2021

@author: deepa
"""

import numpy as np
from scipy.stats import multivariate_normal

""" We define an abstract kernel class for each MCMC method.

    Common methods of this class are -:
        1) pdf - Computes kernel.pdf(x,y) computes p(x,y) for that kernel
        2) rvs - Samples from kernel.rvs(x) samples from p(X,.)
    
"""   

class MRWKernel:
    
    def __init__(self,h):
        self.h = h

    
    def pdf(self,x,y):
        
        if type(x) is np.float64 or type(x) is np.float32:
            dimension = 1
        else:
            dimension = x.shape[0]
        
        cov = 2*self.h*np.eye(dimension)
        
        return multivariate_normal.pdf(x - y,cov=cov)
    
    def rvs(self,x):
                
        if type(x) is np.float64 or type(x) is np.float32:
            dimension = 1
        else:
            dimension = x.shape[0]
        
        cov = 2*self.h*np.eye(dimension)

        return multivariate_normal.rvs(mean=x,cov=cov)


class ULAKernel:
    
    """ Target density is assumed to be exp(-f(x))"""
    
    def __init__(self,h,gradf=None):
        self.h = h
        self.gradf = gradf
    
    def pdf(self,x,y):
        
        if type(x) is np.float64 or type(x) is np.float32:
            dimension = 1
        else:
            dimension = x.shape[0]
        
        cov = 2*self.h*np.eye(dimension)
        
        return multivariate_normal.pdf(x - self.h * self.gradf(x) - y, cov=cov)
    
    
    def rvs(self,x):
        
        if type(x) is np.float64 or type(x) is np.float32:
            dimension = 1
        else:
            dimension = x.shape[0]
        
        cov = 2*self.h*np.eye(dimension)
        
        return multivariate_normal.rvs(mean = x - self.h * self.gradf(x), cov = cov)
    


    


