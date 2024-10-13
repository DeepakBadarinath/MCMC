# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 10:16:11 2021

@author: deepa
"""


import numpy as np
import matplotlib.pyplot as plt
import bisect
from scipy.stats import multivariate_normal

NAME = 'MRW'
#%%


def final_state(initial_distribution, step_size, max_iterations = 10000,
                grad_potential = None, potential = None, target_density = None):
    
    """ 
    Metropolis Random Walk
    
    Metrolpolis Hastings method with chain update
    
    X -> X + root(2step_size)Z
    
    """
    x = initial_distribution.rvs()
    root_two_step_size = np.sqrt(2*step_size)
    dim = x.shape[0]


    

    for i in range(max_iterations):
        
        y = x + root_two_step_size * np.random.randn(dim)
        
        accept_prob = min(1, np.exp(-potential(y) + potential(x)))
        
        U = np.random.rand()
        
        if U < accept_prob:
            
            x = y
    
    return x


#%%
def set_step_size(error_tol, dimension,
                 condition_number, upper_hessian_bound, lower_hessian_bound):
    
    return 1 / (dimension * condition_number * upper_hessian_bound)

#%%
'''
""" 
The target density is a symmetric mixture of a pair of 2D gaussians, with means on (0.5,0.5) and -(0.5,0.5)

Axes of maximal variation is the x=y line and x=-y line
We look at the distribution projected to these axes

"""
a = np.array([0.5,0.5])

target_density = lambda x : 0.5 * 0.5 * 1/np.pi * (np.exp(-np.dot(x-a,x-a)/2) + np.exp(-np.dot(x+a,x+a)/2))

# target density = exp(-f(x)), gradf is the gradient of this f
gradf = lambda x : x - a + 2 * a * 1/(1 + np.exp(2 * np.dot(x,a)))

#Constants bounding the rate of growth of the Hessian, L is upper and m is lower bound
L  = 1
m = 0.5
K = L/m
#%%
#Empirical Sampling
empirical_size = 250000
z = np.random.rand(empirical_size,1) < 0.5

empirical_samples = multivariate_normal.rvs(mean=a,size=empirical_size) * z + (1-z) * multivariate_normal.rvs(mean=-a,size=empirical_size)
empirical_xs = empirical_samples[:,0]
empirical_ys = empirical_samples[:,1]
#%%

#Histogram Test
d = 2
errorTol = 0.2
h = 1 / (d * K * L)
xs = []
ys = []
T_max = 100
noOfSampledPoints = 50000
maxIterations = int(T_max/h)
for i in range(noOfSampledPoints):
    final_state = MRW(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations)
    xs.append(final_state[0])
    ys.append(final_state[1])



#%%
fig,axes = plt.subplots(1,2,figsize=(6,8))
fig.suptitle('MRW-Comparison of histograms along slices')

axes[0].set_title('X axis')
axes[0].hist(xs,bins=100,range=(-5,5),density=True,alpha=0.6,label='MRW')
axes[0].hist(empirical_xs,bins=100,range=(-5,5),density=True,alpha=0.6,label='target')
axes[0].legend()

axes[1].set_title('Y axis')
axes[1].hist(ys,bins=100,range=(-5,5),density=True,alpha=0.6,label='MRW')
axes[1].hist(empirical_ys,bins=100,range=(-5,5),density=True,alpha=0.6,label='target')
axes[1].legend()


plt.savefig("MRW-Comparison of histograms along slices")
plt.show()



#%%
from mpl_toolkits.mplot3d import axes3d

x = np.array(xs)
y = np.array(ys)

counts,xbins,ybins = np.histogram2d(x,y,bins=100, range = [[-5,5],[-5,5]],density=True)
empirical_counts,xbins,ybins = np.histogram2d(empirical_xs,empirical_ys,bins=100, range = [[-5,5],[-5,5]],density=True)
Xbins,Ybins = np.meshgrid(xbins[:-1],ybins[:-1])


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')


# Plot a 3D surface
ax.set_title("MRW 2D Histogram")
ax.plot_surface(Xbins, Ybins, counts,label='MRW',color='b')
ax.plot_surface(Xbins, Ybins, empirical_counts,label='Target',color='r')

#Hack to add legend for surface plots
fake2Dline_ula = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
fake2Dline_target = mpl.lines.Line2D([1],[1], linestyle="none", c='r', marker = 'o')

ax.legend([fake2Dline_ula,fake2Dline_target], ['MRW','Target'], numpoints = 1)

plt.savefig('MRW 2D Histogram')
plt.show()




'''
#%%
def states(initial_distribution, step_size, max_iterations = 10000,
           grad_potential = None, potential = None, target_density = None):
    
    x = initial_distribution.rvs()
    root_two_h = np.sqrt(2*step_size)
    dim = x.shape[0]

    states = [x]
    
    
    for i in range(max_iterations):
        
        y = x + root_two_h * np.random.randn(dim)
        
        accept_prob = min(1, np.exp(-potential(y) + potential(x)))
        
        U = np.random.rand()
        
        if U < accept_prob:
            
            x = y
        
        states.append(x)
    
    return states
#%%
def three_fourth_quantile_mixing(step_size,initial_distribution,
                                 potential, grad_potential, target_density,
                                 target_quantile, mixing_dimension = 0 ,
                                 max_iterations = 10000, error_tol = 0.2):
    
    x = initial_distribution.rvs()
    root_two_step_size = np.sqrt(2*step_size)
    dim = x.shape[0]

    
    sorted_states = [x[mixing_dimension]]
    
    for i in range(max_iterations):
        
        y = x + root_two_step_size * np.random.randn(dim)
        
        accept_prob = min(1, np.exp(-potential(y) + potential(x)))
        
        U = np.random.rand()
        
        if U < accept_prob:
            
            x = y
        
        bisect.insort(sorted_states,x[mixing_dimension])
        
        #len(sorted_mixing_states) = i+2
        if (i+2) % 4 == 0:
            empirical_quantile = sorted_states[int(0.75 * (i+2)) - 1]
        else:
            fraction = 0.75 * (i+2)
            
            #We subract 1 due to 0 indexing
            
            n = int(fraction) - 1
            
            #fractions lies bw n+1 and n+2
            
            empirical_quantile = (sorted_states[n]*(n+2-fraction)
                                  + sorted_states[n+1]*(fraction-n-1))
            
        if np.abs(empirical_quantile - target_quantile) < error_tol:
            return i+1
    
    return i+1
    


#%%
'''
def discrete_tv_error_2d(hist1,hist2):
    return np.sum(np.abs(hist1-hist2)) / (hist1.shape[0] * hist1.shape[1])


empirical_size = 250000
z = np.random.rand(empirical_size,1) < 0.5
empirical_samples = multivariate_normal.rvs(mean=a,size=empirical_size) * z + (1-z) * multivariate_normal.rvs(mean=-a,size=empirical_size)
target_hist,_,_ = np.histogram2d(empirical_samples[:,0],empirical_samples[:,1],bins=100, range = [[-5,5],[-5,5]],density=True)

#%%
#TV error VS iterations 

maxIterations = 700

errorTol = 0.2
h = 1 / (2 * K * L)

tv_errors_samples = []

fig,axes = plt.subplots(1,2,figsize=(6,8))
#fig.suptitle('ULA-TV Errors')

for samples in range(5):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes[0].set_yscale('log')
#plt.xscale('log')
axes[0].plot(range(maxIterations),mean_tv_errors,label = f'MRW h = {np.round(h,5)}')


errorTol = 0.1
h = 0.2*h

tv_errors_samples = []

for samples in range(5):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes[0].set_yscale('log')
#plt.xscale('log')
axes[0].plot(range(maxIterations),mean_tv_errors,label = f'MRW h = {np.round(h,5)}')

errorTol = 1.0
h = 25*h

tv_errors_samples = []

for samples in range(5):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
axes[0].set_yscale('log')
#plt.xscale('log')
axes[0].plot(range(maxIterations),mean_tv_errors,label = f'MRW h = {np.round(h,5)}')

axes[0].legend()
axes[0].set_title("Discrete TV error Vs Iteration")
axes[0].set_ylabel("TV error")
axes[0].set_xlabel("Iteration")

T_max = 15
errorTol = 0.2
h = 1 / (2 * K * L)
maxIterations = int(T_max / h)

tv_errors_samples = []

for samples in range(5):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))
    
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)

axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'MRW h = {np.round(h,5)}')



errorTol = 0.1
h = 0.2*h
maxIterations = int(T_max / h)

tv_errors_samples = []

for samples in range(5):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))
    
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)

axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'MRW h = {np.round(h,5)}')


errorTol = 1.0
h = 25*h
maxIterations = int(T_max / h)

tv_errors_samples = []

for samples in range(5):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = maxIterations))
    
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)

axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'MRW h = {np.round(h,5)}')

#axes[1].legend()
axes[1].set_title("Discrete TV error Vs Time")
axes[1].set_ylabel("TV error")
axes[1].set_xlabel("Time")

fig.tight_layout()

plt.savefig("MRW_TVErrors")

#plt.savefig("ULA_no2 Tv error vs time")
plt.show()
#%%
#Discrete TV error

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
maxIterations = 500

h = 0.2 / (2 * K * L)

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

plt.plot(range(maxIterations),mean_tv_errors,label = f'MRW h = {h}')


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

plt.plot(range(maxIterations),mean_tv_errors,label = f'MRW h = {h}')

h = 2.5 / (2 * K * L)

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

plt.plot(range(maxIterations),mean_tv_errors,label = f'MRW h = {h}')

plt.legend()
plt.title("MRW TV error Vs Iteration")
plt.savefig("MRW_TVError_Iteration")
plt.show()

#%%

##TV error vs Time
T_max = 30

h = 5 / (2 * K * L)

maxIterations = int(T_max / h)

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
        tv_errors.append((error1 + error2))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')

plt.plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'MRW h = {h}')


h = 1 / (2 * K * L)

maxIterations = int(T_max / h)

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
        tv_errors.append((error1 + error2))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')

plt.plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'MRW h = {h}')


h = 0.2 / (2 * K * L)

maxIterations = int(T_max / h)

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
        tv_errors.append((error1 + error2))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')

plt.plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'MRW h = {h}')

plt.legend()

plt.savefig("MRW Tv error vs time")
plt.show()
#%%
import bisect

def MRWMixing(target_density,h,initial_distribution,
              quantile, target_quantile, mixing_dimension = 0,
              maxIterations = 10000, errorTol = 0.2):
    
    x = initial_distribution.rvs()
    root_two_h = np.sqrt(2*h)
    dim = x.shape[0]

    sorted_mixing_states = [x[mixing_dimension]]
    
    
    for i in range(maxIterations):
        
        y = x + root_two_h * np.random.randn(dim)
        
        accept_prob = min(1, target_density(y) / target_density(x) )
        
        U = np.random.rand()
        
        if U < accept_prob:
            
            x = y
        
        bisect.insort(sorted_mixing_states,x[mixing_dimension])
        
        #len(sorted_mixing_states) = i+2
        if (i+2) % 4 == 0:
            empirical_quantile = sorted_mixing_states[int(quantile * (i+2)) - 1]
        else:
            empirical_quantile = sorted_mixing_states[int(quantile * (i+2))]
            
        if np.abs(empirical_quantile - target_quantile) < errorTol:
            return i+1
    
    return i+1


#%%
#T_mix and dimension

import scipy.stats

errorTol = 0.2
m = 1/4
L = 1
K = L/m

quantile = 0.75
target_quantile = scipy.stats.norm.ppf(quantile,scale=2)


mixing_times = []
dimensions =  np.arange(5,16)

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


#%%
plt.yscale('log')
plt.xscale('log')
plt.plot(dimensions,mixing_times,'og')
plt.savefig('tmix vs dim loglog - MRW')  
plt.show()
plt.plot(dimensions,mixing_times,'og')  
plt.savefig('tmix vs dim - MRW')  
plt.show()
#%%
m,b = np.polyfit(dimensions,mixing_times,1)
plt.title('MRW tmix 0.75 quantile - tmix vs dim')
plt.plot(dimensions,m*dimensions + b)
plt.plot(dimensions,mixing_times,'og')
plt.xlabel('Dimension')
plt.ylabel('Tmix')
plt.savefig('MRW tmix vs dim - linear')
plt.show()
m
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
plt.title('MRW - tmix vs dim')
plt.savefig('MRW-tmix vs dim')

#%%
popt
#%%

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(np.log10(np.arange(5,16)), np.log10(mixing_times))


xfid = np.linspace(0.7,1.3)     # This is just a set of x to plot the straight line 

plt.plot(np.log10(dimensions), np.log10(mixing_times), 'k.')
plt.plot(xfid, xfid*slope+intercept)
plt.show()

slope
#%%
#Tmix vs 1/delta
import scipy.stats
d = 15

m = 1/4
L = 1
K = L/m

quantile = 0.75
target_quantile = scipy.stats.norm.ppf(quantile,scale=2)

diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
sigma_inv = np.diag(1 / diagonal_entries)
target_density = lambda x : np.power(2 * np.pi,-d/2) * np.exp( -(x.T @ sigma_inv @ x) / 2 )

mixing_times = []
invErrorTols = np.arange(2,10)

for invErrorTol in invErrorTols:
    errorTol = 1 / invErrorTol
    h = 1 / (d * K * L)
    
    mixing_time_sum = 0
    
    for i in range(200):
        
        mixing_time_sum += MRWMixing(target_density,h,multivariate_normal(cov= 1/L * np.eye(d)),
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 10000)
        
    mixing_time_avg = mixing_time_sum / 200
    
    mixing_times.append(mixing_time_avg)

#%%
plt.xscale('log')
plt.yscale('log')
plt.plot(invErrorTols,mixing_times,'og')
plt.savefig('tmix  vs delta, d = 15, MRW')
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
plt.savefig('tmix vs delta loglog d=15 MRW')
plt.plot(newX, myExpFunc(newX, *popt), 'b')
#%%
popt
#%%
plt.plot(range(2,10),mixing_times,'og')
#%%
#Logistic Regression - No preconditioning 
n = 50
d = 2
X = 2 * (np.random.rand(n,d)>0.5) - 1
X = X / np.sqrt(d)

theta_true = np.ones((d,1))
y = np.exp(X @ theta_true) / (1 + np.exp(X @ theta_true))

sigma_x = 1/n * X.T @ X
alpha = 0.1


def f(theta):
    theta_column = theta.reshape((-1,1))
    return (-y.T @ X @ theta_column + alpha * theta.T @ sigma_x @ theta_column + np.sum(np.log(1 + np.exp(X @ theta_column))))[0,0]

gradf_constant = -X.T @ y 

def gradf(theta):
    theta_column = theta.reshape((-1,1))
    return np.squeeze(gradf_constant + alpha * sigma_x @ theta_column + (np.sum(X/(1 + np.exp(-X @ theta_column)),axis=0).reshape((-1,1))))

target_density = lambda x : np.exp(-f(x))
#%%
eigvals = np.linalg.eigvalsh(sigma_x)
L = (0.25 * n + alpha) * eigvals[-1]
m = alpha * eigvals[0]

#%%
maxIterations = 4000
h = 1 / (2 * K * L)

l1_errors_samples = []

for samples in range(100):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations))
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/2
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'MRW h = {np.round(h,5)}')



h = 0.2 / (2 * K * L)

l1_errors_samples = []

for samples in range(100):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations))
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/2
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'MRW h = {np.round(h,5)}')


h = 5 / (2 * K * L)

l1_errors_samples = []

for samples in range(100):
    states = np.array(MRW_states(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations))
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/2
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'MRW h = {np.round(h,5)}')

plt.legend()
plt.title('MRW - Mean l1 error- Logistic regression')
plt.ylabel('l1 Error')
plt.xlabel('Iteration')
plt.savefig('MRW-Mean l1 error, no pre conditioning')
#%%
'''