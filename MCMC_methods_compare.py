# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:34:06 2021

@author: deepa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#%%
from MALA import *
from ULA import *
from MRW import *

#%%
""" 
The target density is a symmetric mixture of a pair of 2D gaussians, with means on (0.5,0.5) and -(0.5,0.5)

Axes of maximal variation is the x=y line and x=-y line
We look at the distribution projected to these axes

"""
a = np.array([0.5,0.5])

target_density = lambda x : 0.5 * (multivariate_normal.pdf(x,mean=a) + multivariate_normal.pdf(x,mean=-a))
f = lambda x : 0.5 * np.dot(x-a,x-a) - np.log(1+np.exp(-2*np.dot(x,a)))

# target density = exp(-f(x)), gradf is the gradient of this f
gradf = lambda x : x - a + 2 * a * 1/(1 + np.exp(2 * np.dot(x,a)))

#Constants bounding the rate of growth of the Hessian, L is upper and m is lower bound
L  = 1
m = 0.5
K = L/m



h_MRW = 1 / (2 * K * L)
errorTol = 0.2
h_ULA = (errorTol**2) / (2 * K * L)
h_MALA = 1 / L * (min(1 / np.sqrt(2 * K), 1 / 2))

#%%

def discrete_tv_error_2d(hist1,hist2):
    return np.sum(np.abs(hist1-hist2)) / (hist1.shape[0] * hist1.shape[1])


empirical_size = 250000
z = np.random.rand(empirical_size,1) < 0.5
empirical_samples = multivariate_normal.rvs(mean=a,size=empirical_size) * z + (1-z) * multivariate_normal.rvs(mean=-a,size=empirical_size)
target_hist,_,_ = np.histogram2d(empirical_samples[:,0],empirical_samples[:,1],bins=100, range = [[-5,5],[-5,5]],density=True)
#%%
maxIterations = 700

tv_errors_samples = []

fig,axes = plt.subplots(1,1)
#fig.suptitle('ULA-TV Errors')

for samples in range(5):
    states = np.array(MALA_states(gradf,f,h_MALA,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes.set_yscale('log')
#plt.xscale('log')
axes.plot(range(maxIterations),mean_tv_errors,label = f'MALA h = {np.round(h_MALA,5)}')

tv_errors_samples = []

for samples in range(5):
    states = np.array(MRW_states(target_density,h_MRW,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes.set_yscale('log')
#plt.xscale('log')
axes.plot(range(maxIterations),mean_tv_errors,label = f'MRW h = {np.round(h_MRW,5)}')

tv_errors_samples = []

for samples in range(5):
    states = np.array(ULA_states(h_ULA,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes.set_yscale('log')
#plt.xscale('log')
axes.plot(range(maxIterations),mean_tv_errors,label = f'ULA h = {np.round(h_ULA,5)}')

axes.set_title("Discrete TV error Vs Iterations")
axes.set_ylabel("TV error")
axes.set_xlabel("Iteration")

axes.legend()

'''
T_max = 15
maxIterations = int(T_max / h_MALA)
tv_errors_samples = []



for samples in range(5):
    states = np.array(MALA_states(gradf,f,h_MALA,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'MALA h = {np.round(h_MALA,5)}')

maxIterations = int(T_max / h_MRW)
tv_errors_samples = []

for samples in range(5):
    states = np.array(MRW_states(target_density,h_MRW,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'MRW h = {np.round(h,5)}')


maxIterations = int(T_max / h_ULA)
tv_errors_samples = []

for samples in range(5):
    states = np.array(ULA_states(h_ULA,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'ULA h = {np.round(h,5)}')


axes[1].set_title("Discrete TV error Vs Time")
axes[1].set_ylabel("TV error")
axes[1].set_xlabel("Time")
'''
fig.tight_layout()

plt.savefig("TVErrors")

#plt.savefig("ULA_no2 Tv error vs time")
plt.show()

#%%
def discrete_tv_error(hist1,hist2):
    return np.sum(np.abs(hist1-hist2)) / hist1.shape[0]


empirical_size = 250000
z = np.random.rand(empirical_size,1) < 0.5
empirical_samples = multivariate_normal.rvs(mean=a,size=empirical_size) * z + (1-z) * multivariate_normal.rvs(mean=-a,size=empirical_size)
target_first_principal_axis = np.linalg.norm(empirical_samples @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(empirical_samples) @ np.array([1,1]))
target_second_principal_axis = np.linalg.norm(empirical_samples @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(empirical_samples) @ np.array([1,-1]))

target_first_hist,_ = np.histogram(target_first_principal_axis, bins=100, range=(-5.2,5.2), density=True)
target_second_hist,_ = np.histogram(target_second_principal_axis, bins=100, range=(-5.2,5.2), density=True)
#%%

#TV error vs iteration
maxIterations = 700

d = 2
h = 1/L * min(1/d, 1/np.sqrt(d*K))

tv_errors_samples = []

for samples in range(20):
    states = MALA_states(gradf,f,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations)
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
    second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist1,_ = np.histogram(first_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
        hist2,_ = np.histogram(second_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
        error1 = discrete_tv_error(hist1,target_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append((error1 + error2)/2)
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')

plt.plot(range(maxIterations),mean_tv_errors,label = f'MALA')

h = 1 / (2 * K * L)

tv_errors_samples = []

for samples in range(20):
    states = MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations)
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
    second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist1,_ = np.histogram(first_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
        hist2,_ = np.histogram(second_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
        error1 = discrete_tv_error(hist1,target_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append((error1 + error2)/2)
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')

plt.plot(range(maxIterations),mean_tv_errors,label = f'MRW')

errorTol = 0.2
h = (errorTol**2) / (2 * K * L)

tv_errors_samples = []

for samples in range(20):
    states = ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf)
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
    second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist1,_ = np.histogram(first_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
        hist2,_ = np.histogram(second_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
        error1 = discrete_tv_error(hist1,target_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append((error1 + error2)/2)
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')

plt.plot(range(maxIterations),mean_tv_errors,label = f'ULA')
plt.legend()
plt.title("MCMC methods TV error Vs Iteration")
plt.savefig("MCMC_TVError_Iteration")
plt.show()

#%%

from statsmodels.graphics.tsaplots import plot_acf


h_MRW = 1 / (2 * K * L)
errorTol = 0.2
h_ULA = (errorTol**2) / (2 * K * L)
h_MALA = 1 / L * (min(1 / np.sqrt(2 * K), 1 / 2))


fig,axes = plt.subplots()
axes.set_title("Autocorrelation Plots for various MCMC methods")
axes.set_xlabel("Lag")
axes.set_ylabel("Autocorrelation")

sample_states = []

states = MALA_states(gradf,f,h_MALA,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations)
first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

kwargs = {'label':'MALA'}
plot_acf(first_principal_axis, ax = axes ,**kwargs)
axes.legend()


states = MRW_states(target_density,h_MRW,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations)
first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

#kwargs = {'label':'MRW','alpha': 0.2,'color':'r','lw':.1}
kwargs = {'label':'MRW'}
plot_acf(first_principal_axis, ax = axes , **kwargs)

states = ULA_states(h_ULA,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf)
first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

#kwargs = {'label':'ULA','alpha': 0.2,'color':'g','lw':0.1}
kwargs = {'label':'ULA'}
plot_acf(first_principal_axis, ax = axes , **kwargs)

axes.legend()
plt.savefig("Autocorrelation of different MCMC methods-1")
plt.show()


#%%

#T_mix and dimension

import scipy.stats

errorTol = 0.2
m = 1/4
L = 1
K = L/m

quantile = 0.75
#target_quantile = scipy.stats.norm.ppf(quantile,scale=np.sqrt(2))
target_quantile = scipy.stats.norm.ppf(quantile,scale=2)
#%%

mixing_times = []
dimensions = np.arange(2,20)


for d in dimensions:
    h = (errorTol**2) / (d * K * L)
    
    #diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
    diagonal_entries = np.array([4.0] + list(np.ones(d-1)))

    gradf = lambda x : np.diag(1 / diagonal_entries) @ x
    
    mixing_time_sum = 0
    
    for i in range(1000):
        
        mixing_time_sum += ULAMixing(h,multivariate_normal(cov= 1/L * np.eye(d)),gradf,
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 10000)
        
    mixing_time_avg = mixing_time_sum / 1000
    
    mixing_times.append(mixing_time_avg)

mixing_times_ULA = mixing_times

#%%

mixing_times = []

for d in dimensions:
    h = 1 / (d * K * L)
    
    #diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
    diagonal_entries = np.array([4.0] + list(np.ones(d-1)))
    
    sigma_inv = np.diag(1 / diagonal_entries)

    target_density = lambda x : np.power(2 * np.pi,-d/2) * np.exp( -(x.T @ sigma_inv @ x) / 2 )
    
    mixing_time_sum = 0
    
    for i in range(1000):
        
        mixing_time_sum += MRWMixing(target_density,h,multivariate_normal(cov= 1/L * np.eye(d)),
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 10000)
        
    mixing_time_avg = mixing_time_sum / 1000
    
    mixing_times.append(mixing_time_avg)
    
mixing_times_MRW = mixing_times

#%%

mixing_times = []



for d in dimensions:
    h = 1/L * min(1/d, 1/np.sqrt(d*K))
    
    #diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
    diagonal_entries = np.array([4.0] + list(np.ones(d-1)))
    
    sigma_inv = np.diag(1 / diagonal_entries)

    gradf = lambda x : sigma_inv @ x
    
    f = lambda x : 0.5 * x.T @ sigma_inv @ x
    
    mixing_time_sum = 0
    
    for i in range(1000):
        
        mixing_time_sum += MALAMixing(gradf,f,h,multivariate_normal(cov= 1/L * np.eye(d)),
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 10000)
        
    mixing_time_avg = mixing_time_sum / 1000
    
    mixing_times.append(mixing_time_avg)

    
mixing_times_MALA = mixing_times

#%%
linslopes = {}
m,b = np.polyfit(dimensions,mixing_times_ULA,1)
plt.plot(dimensions,m*dimensions + b,c='g',label='ULA')
plt.plot(dimensions,mixing_times_ULA,'og')
linslopes['ULA'] = m

m,b = np.polyfit(dimensions,mixing_times_MRW,1)
plt.plot(dimensions,m*dimensions + b,c='r',label='MRW')
plt.plot(dimensions,mixing_times_MRW,'or')
linslopes['MRW'] = m

m,b = np.polyfit(dimensions,mixing_times_MALA,1)
plt.plot(dimensions,m*dimensions + b,c='b',label='MALA')
plt.plot(dimensions,mixing_times_MALA,'ob')
linslopes['MALA'] = m

plt.xlabel('Dimension')
plt.ylabel('Tmix')
plt.title('Tmix vs Dimension - linear scale')
plt.legend()
plt.savefig('Tmix vs dim - linear')
plt.show()
#%%
def myExpFunc(x, a, b):
    return a * np.power(x, b)

#%%
    

from scipy.optimize import curve_fit
fig = plt.figure()
ax=plt.gca()

ax.scatter(dimensions,mixing_times_ULA,c="green",alpha=0.95,edgecolors='none')
ax.set_yscale('log')
ax.set_xscale('log')

newX = np.logspace(0.25, 1.3, base=10)  # Makes a nice domain for the fitted curves.
                                   # Goes from 10^0 to 10^3
                                   # This avoids the sorting and the swarm of lines.

popt, pcov = curve_fit(myExpFunc, dimensions, mixing_times_ULA)
plt.plot(newX, myExpFunc(newX, *popt), 'g',label='ULA')


ax.scatter(dimensions,mixing_times_MRW,c="red",alpha=0.95,edgecolors='none')
popt, pcov = curve_fit(myExpFunc, dimensions, mixing_times_MRW)
plt.plot(newX, myExpFunc(newX, *popt), 'r',label='MRW')

ax.scatter(dimensions,mixing_times_MALA,c="blue",alpha=0.95,edgecolors='none')
popt, pcov = curve_fit(myExpFunc, dimensions, mixing_times_MALA)
plt.plot(newX, myExpFunc(newX, *popt), 'b',label='MALA')


plt.legend()
plt.title('Tmix vs dim - loglog scale')
plt.xlabel('Dimension')
plt.ylabel('Tmix')

plt.savefig('tmix vs dim - loglog scale')
#%%
logslopes = {}
plt.xscale('log')
plt.yscale('log')

newX = np.logspace(0, 2, base=10)

m,b = np.polyfit(np.log(dimensions),np.log(mixing_times_ULA),1)
plt.plot(newX,myExpFunc(newX,b,m),c='g',label='ULA')
plt.plot(dimensions,mixing_times_ULA,'og')
logslopes['ULA'] = m

m,b = np.polyfit(np.log(dimensions),np.log(mixing_times_MRW),1)
plt.plot(newX,myExpFunc(newX,b,m),c='r',label='MRW')
plt.plot(dimensions,mixing_times_MRW,'or')
logslopes['MRW'] = m

m,b = np.polyfit(np.log(dimensions),np.log(mixing_times_MALA),1)
plt.plot(newX,myExpFunc(newX,b,m),c='b',label='MALA')
plt.plot(dimensions,mixing_times_MALA,'ob')
logslopes['MALA'] = m

plt.xlabel('Dimension')
plt.ylabel('Tmix')
plt.title('Tmix vs Dimension - loglog scale')
plt.legend()
plt.savefig('Tmix vs dim - log')
plt.show()
#%%
logslopes

#%%
from scipy.optimize import curve_fit
fig = plt.figure()
ax=plt.gca() 
ax.scatter(dimensions,mixing_times,c="blue",alpha=0.95,edgecolors='none')
ax.set_yscale('log')
ax.set_xscale('log')

newX = np.logspace(0.7, 1.3, base=10)  # Makes a nice domain for the fitted curves.
                                   # Goes from 10^0 to 10^3
                                   # This avoids the sorting and the swarm of lines.

# Let's fit an exponential function.  
# This looks like a line on a lof-log plot.
def myExpFunc(x, a, b):
    return a * np.power(x, b)
popt, pcov = curve_fit(myExpFunc, dimensions, mixing_times)
plt.plot(newX, myExpFunc(newX, *popt), 'b')
plt.title('MALA - tmix vs dim')
plt.savefig('MALA-tmix vs dim')


#%%
invTols = np.arange(2,10)
#%%
#Tmix vs 1/delta
d = 5
diagonal_entries = np.array([4.0] + list(np.ones(d-1)))
gradf = lambda x : np.diag(1 / diagonal_entries) @ x
mixing_times = []
invTols = np.arange(2,10)

for invErrorTol in range(2,10):
    errorTol = 1 / invErrorTol
    h = (errorTol**2) / (d * K * L)
    
    mixing_time_sum = 0
    
    for i in range(500):
        
        mixing_time_sum += ULAMixing(h,multivariate_normal(cov= 1/L * np.eye(d)),gradf,
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 11000)
        
    mixing_time_avg = mixing_time_sum / 500
    
    mixing_times.append(mixing_time_avg)


mixing_times_ULA = mixing_times


#%%
#MRW 
sigma_inv = np.diag(1 / diagonal_entries)
target_density = lambda x : np.power(2 * np.pi,-d/2) * np.exp( -(x.T @ sigma_inv @ x) / 2 )

mixing_times = []

for invErrorTol in invTols:
    errorTol = 1 / invErrorTol
    h = 1 / (d * K * L)
    
    mixing_time_sum = 0
    
    for i in range(500):
        
        mixing_time_sum += MRWMixing(target_density,h,multivariate_normal(cov= 1/L * np.eye(d)),
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 11000)
        
    mixing_time_avg = mixing_time_sum / 500
    
    mixing_times.append(mixing_time_avg)
    
mixing_times_MRW = mixing_times

#%%

d = 5

diagonal_entries = np.array([4.0] + list(np.ones(d-1)))
    
sigma_inv = np.diag(1 / diagonal_entries)

gradf = lambda x : sigma_inv @ x

f = lambda x : 0.5 * x.T @ sigma_inv @ x 

mixing_times = []

for invErrorTol in invTols:
    errorTol = 1 / invErrorTol
    
    h = 1/L * min(1/d, 1/np.sqrt(d*K))
    
    mixing_time_sum = 0
    
    for i in range(500):
        
        mixing_time_sum += MALAMixing(gradf,f,h,multivariate_normal(cov= 1/L * np.eye(d)),
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 10000)
        
    mixing_time_avg = mixing_time_sum / 500
    
    mixing_times.append(mixing_time_avg)

mixing_times_MALA = mixing_times

#%%
plt.xscale('log')
plt.yscale('log')
plt.plot(invTols,mixing_times,'og')
#plt.savefig('tmix  vs delta, d = 15')
plt.show()
 
    
#%%
from scipy.optimize import curve_fit
fig = plt.figure()
ax=plt.gca() 
ax.scatter(np.arange(2,10),mixing_times,c="blue",alpha=0.95,edgecolors='none')
ax.set_yscale('log')
ax.set_xscale('log')

newX = np.logspace(0.2, 1, base=10)  # Makes a nice domain for the fitted curves.
                                   # Goes from 10^0 to 10^3
                                   # This avoids the sorting and the swarm of lines.

# Let's fit an exponential function.  
# This looks like a line on a lof-log plot.
def myExpFunc(x, a, b):
    return a * np.power(x, b)
popt, pcov = curve_fit(myExpFunc, np.arange(2,10), mixing_times)
plt.plot(newX, myExpFunc(newX, *popt), 'b')
#plt.savefig('tmix  vs delta, d = 5, 100runs')
#%%
delslope = np.polyfit(np.log(range(2,10)),np.log(mixing_times),1)
delslope

#%%
linslopes = {}
dimensions = invTols
m,b = np.polyfit(dimensions,mixing_times_ULA,1)
plt.plot(dimensions,m*dimensions + b,c='g',label='ULA')
plt.plot(dimensions,mixing_times_ULA,'og')
linslopes['ULA'] = m

m,b = np.polyfit(dimensions,mixing_times_MRW,1)
plt.plot(dimensions,m*dimensions + b,c='r',label='MRW')
plt.plot(dimensions,mixing_times_MRW,'or')
linslopes['MRW'] = m

m,b = np.polyfit(dimensions,mixing_times_MALA,1)
plt.plot(dimensions,m*dimensions + b,c='b',label='MALA')
plt.plot(dimensions,mixing_times_MALA,'ob')
linslopes['MALA'] = m

plt.xlabel('1/delta')
plt.ylabel('Tmix')
plt.title('Tmix vs 1/delta - linear scale, dim = 5')
plt.legend()
plt.savefig('Tmix vs invdelta - linear')
plt.show()
#%%
dimensions = invTols
from scipy.optimize import curve_fit
fig = plt.figure()
ax=plt.gca()

ax.scatter(dimensions,mixing_times_ULA,c="green",alpha=0.95,edgecolors='none')
ax.set_yscale('log')
ax.set_xscale('log')

newX = np.logspace(0.25, 1, base=10)  # Makes a nice domain for the fitted curves.
                                   # Goes from 10^0 to 10^3
                                   # This avoids the sorting and the swarm of lines.

popt, pcov = curve_fit(myExpFunc, dimensions, mixing_times_ULA)
plt.plot(newX, myExpFunc(newX, *popt), 'g',label='ULA')


ax.scatter(dimensions,mixing_times_MRW,c="red",alpha=0.95,edgecolors='none')
popt, pcov = curve_fit(myExpFunc, dimensions, mixing_times_MRW)
plt.plot(newX, myExpFunc(newX, *popt), 'r',label='MRW')

ax.scatter(dimensions,mixing_times_MALA,c="blue",alpha=0.95,edgecolors='none')
popt, pcov = curve_fit(myExpFunc, dimensions, mixing_times_MALA)
plt.plot(newX, myExpFunc(newX, *popt), 'b',label='MALA')


plt.legend()
plt.title('Tmix vs 1/delta - loglog scale,dim = 5')
plt.xlabel('1/delta')
plt.ylabel('Tmix')

plt.savefig('tmix vs invdelta - loglog')
#%%
logslopes = {}
plt.xscale('log')
plt.yscale('log')

newX = np.logspace(0, 2, base=10)

m,b = np.polyfit(np.log(dimensions),np.log(mixing_times_ULA),1)
plt.plot(newX,myExpFunc(newX,b,m),c='g',label='ULA')
plt.plot(dimensions,mixing_times_ULA,'og')
logslopes['ULA'] = m

m,b = np.polyfit(np.log(dimensions),np.log(mixing_times_MRW),1)
plt.plot(newX,myExpFunc(newX,b,m),c='r',label='MRW')
plt.plot(dimensions,mixing_times_MRW,'or')
logslopes['MRW'] = m

m,b = np.polyfit(np.log(dimensions),np.log(mixing_times_MALA),1)
plt.plot(newX,myExpFunc(newX,b,m),c='b',label='MALA')
plt.plot(dimensions,mixing_times_MALA,'ob')
logslopes['MALA'] = m

plt.xlabel('Dimension')
plt.ylabel('Tmix')
plt.title('Tmix vs Dimension - loglog scale')
plt.legend()
plt.savefig('Tmix vs dim - log')
plt.show()
#%%
logslopes