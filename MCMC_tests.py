# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 07:07:28 2021

@author: deepa
"""

import numpy as np
import matplotlib.pyplot as plt


from MCMC_methods import *


    
#%%
""" MRW Mixing Times vs Dimension """

"""We sample 1000 points and take 1000 iterations to sample each point 
    We take initial_distribution = standard normal with mean 0 and cov 0.5 
    We vary the values of h in [0.2,0.4,0.6,0.8]"""
import scipy.stats

errorTol = 0.2
m = 1
L = 4
K = L/m

quantile = 0.75
target_quantile = L * scipy.stats.norm.ppf(quantile)

mixing_times = []
dimensions = []

for d in range(5,16):
    h = 1 / (d * K * L)
    diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
    cov = np.diag(diagonal_entries)
    
    target_density = lambda x : multivariate_normal.pdf(x,cov = cov)
    
    mixing_time_sum = 0
    
    for i in range(10):
        
        mixing_time_sum += MRWMixing(target_density,h,multivariate_normal(cov= 1/L * np.eye(d)),
                                      quantile,target_quantile,
                                      maxIterations = 10000,errorTol=errorTol)
        
    mixing_time_avg = mixing_time_sum / 10
    
    mixing_times.append(mixing_time_avg)

plt.plot(range(5,16),mixing_times,'og')  

    
     

#%%

"""
MALA Mixing times vs dimension

"""

errorTol = 0.2
m = 1
L = 4
K = L/m

quantile = 0.75
target_quantile = L * scipy.stats.norm.ppf(quantile)

mixing_times = []
dimensions = []


for d in range(5,16):
    h = 1 / L * (min(1 / np.sqrt(d * K), 1 / d))
    
    diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
    cov = np.diag(diagonal_entries)
    
    gradf = lambda x : np.diag(1 / diagonal_entries) @ x
    
    target_density = lambda x : multivariate_normal.pdf(x,cov = cov)
    
    mixing_time_sum = 0
    
    for i in range(10):
        
        mixing_time_sum += MALAMixing(target_density,h,multivariate_normal(cov= 1/L * np.eye(d)),gradf,
                                      quantile,target_quantile,errorTol=errorTol)
        
    mixing_time_avg = mixing_time_sum / 10
    
    mixing_times.append(mixing_time_avg)

plt.plot(range(5,16),mixing_times,'og')  
#%%
"""
ULA Mixing times vs dimension

"""
import scipy.stats

errorTol = 1.5
m = 1
L = 4
K = L/m

quantile = 0.75
target_quantile = L * scipy.stats.norm.ppf(quantile)

mixing_times = []
dimensions = []


for d in range(5,16):
    h = (errorTol**2) / (d * K * L)
    
    diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
    cov = np.diag(diagonal_entries)
    
    gradf = lambda x : np.diag(1 / diagonal_entries) @ x
    
    target_density = lambda x : multivariate_normal.pdf(x,cov = cov)
    
    mixing_time_sum = 0
    
    for i in range(10):
        
        mixing_time_sum += ULAMixing(target_density,h,multivariate_normal(cov= 1/L * np.eye(d)),gradf,
                                      quantile,target_quantile,
                                      errorTol=errorTol)
        
    mixing_time_avg = mixing_time_sum / 5
    
    mixing_times.append(mixing_time_avg)

plt.plot(range(5,16),mixing_times,'og')  
#%%


def discrete_tv_error(hist1,hist2):
    return np.sum(np.abs(hist1-hist2))


empirical_size = 250000
empirical_samples = multivariate_normal.rvs(mean=0,cov=4,size=empirical_size)


target_first_hist,_ = np.histogram(empirical_samples, bins=150, range=(-10.2,10.2), density=True)
#%%
_,states = ULA(target_density,h,multivariate_normal(cov= 1/L * np.eye(d)),gradf = gradf)
states.shape

#%%
errorTol = 0.2
m = 1
L = 4
K = L/m


mixing_times = []
dimensions = []


for d in range(5,16):
    h = (errorTol**2) / (d * K * L)
    diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
    cov = np.diag(diagonal_entries)
    
    gradf = lambda x : np.diag(1 / diagonal_entries) @ x
    
    target_density = lambda x : multivariate_normal.pdf(x,cov = cov)
    
    mixing_time_sum = 0
    
    for i in range(2):
        
        _,states = ULA(target_density,h,multivariate_normal(cov= 1/L * np.eye(d)),gradf = gradf)
        #mixing_time = 10001
        
        #for time in range(10001):
            
        hist_sample,_ = np.histogram(np.array(states)[0,:10001],bins=150,range=(-10.2,10.2),density=True)
        error = discrete_tv_error(hist_sample,target_first_hist)
            
        """if error < errorTol:
                mixing_time = time + 1
                print('haha reached lesser tol')
                break"""
        
        print(error)
        #mixing_time_sum += mixing_time
                
    mixing_time_avg = mixing_time_sum / 2
    mixing_times.append(mixing_time_avg)

#plt.plot(range(5,16),mixing_times,'og')  


mixing_times




