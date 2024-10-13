# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:22:00 2021

@author: deepa
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import bisect

NAME = 'ULA'


#%%
def final_state(initial_distribution, step_size, max_iterations = 10000,
                grad_potential = None, potential = None, target_density = None):
    
    """
    Underdamped Langevin Algorithm 
    
    Target density is given by exp(-f(x))
    
    Chain Update:
    X -> X + root(2h)Z - h * gradf(X)
    
    
    
    """
    root_two_h = np.sqrt(2*step_size)
    x = initial_distribution.rvs()
    dim = x.shape[0]
    
    for i in range(max_iterations):
        
        x = x + root_two_h * np.random.randn(dim) - step_size * grad_potential(x)

    
    return x

#%%
def set_step_size(error_tol, dimension,
                 condition_number, upper_hessian_bound, lower_hessian_bound):
#    return error_tol/2 / (dimension * condition_number * upper_hessian_bound)
#    return error_tol*5 / (np.sqrt(dimension) * condition_number * upper_hessian_bound)
    return error_tol/2 / (np.sqrt(dimension) * condition_number * upper_hessian_bound)


#%%
'''
def ULA(h,initial_distribution ,maxIterations = 10000,gradf=None):
    
    """
    Underdamped Langevin Algorithm 
    
    Target density is given by exp(-f(x))
    
    Chain Update:
    X -> X + root(2h)Z - h * gradf(X)
    
    
    
    """
    root_two_h = np.sqrt(2*h)
    x = initial_distribution.rvs()
    dim = x.shape[0]
    
    for i in range(maxIterations):
        
        x = x + root_two_h * np.random.randn(dim) - h * gradf(x)

    
    return x
'''
#%%

""" 
The target density is a symmetric mixture of a pair of 2D gaussians, with means on (0.5,0.5) and -(0.5,0.5)

Axes of maximal variation is the x=y line and x=-y line
We look at the distribution projected to these axes

"""
'''
a = np.array([0.5,0.5])

target_density = lambda x : 0.5 * (multivariate_normal.pdf(x,mean=a) + multivariate_normal.pdf(x,mean=-a))

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

prj1 = np.linalg.norm(empirical_samples @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(empirical_samples) @ np.array([1,1]))
prj2 = np.linalg.norm(empirical_samples @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(empirical_samples) @ np.array([1,-1]))

empirical_xs = empirical_samples[:,0]
empirical_ys = empirical_samples[:,1]

target_first_hist,_ = np.histogram(prj1,bins=100,range=[-5,5])
target_second_hist,_ = np.histogram(prj2,bins=100,range=[-5,5])
'''
#%%

def states(initial_distribution, step_size, max_iterations = 10000,
           grad_potential = None, potential = None, target_density = None):
    
    """
    Underdamped Langevin Algorithm 
    
    Target density is given by exp(-f(x))
    
    Chain Update:
    X -> X + root(2h)Z - step_size * gradf(X)
    
    
    
    """
    root_two_h = np.sqrt(2*step_size)
    x = initial_distribution.rvs()
    dim = x.shape[0]
    states = [x]
    
    for i in range(max_iterations):
        
        x = x + root_two_h * np.random.randn(dim) - step_size * grad_potential(x)
        states.append(x)

    
    return np.array(states)


#%%
    

def three_fourth_quantile_mixing(h,initial_distribution,
                                 potential, grad_potential, target_density,
                                 target_quantile, mixing_dimension = 0 ,
                                 max_iterations = 10000, error_tol = 0.2):
    
    x = initial_distribution.rvs()
    root_two_h = np.sqrt(2*h)
    dim = x.shape[0]

    
    sorted_states = [x[mixing_dimension]]
    
    for i in range(max_iterations):
        
        x = x + (root_two_h * np.random.randn(dim)) - (h * grad_potential(x))
        
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

a = np.array([0.5,0.5])

empirical_size = 250000
z = np.random.rand(empirical_size,1) < 0.5
empirical_samples = multivariate_normal.rvs(mean=a,size=empirical_size) * z + (1-z) * multivariate_normal.rvs(mean=-a,size=empirical_size)
target_hist,_,_ = np.histogram2d(empirical_samples[:,0],empirical_samples[:,1],bins=10, range = [[-5,5],[-5,5]],density=True)


#%%

#TV errors joint graph

maxIterations = 700

errorTol = 0.2
h = (errorTol**2) / (2 * K * L)

tv_errors_samples = []

fig,axes = plt.subplots(1,2,figsize=(6,8))
#fig.suptitle('ULA-TV Errors')

for samples in range(5):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_m = np.mean(np.array(tv_errors_samples),axis = 0)
axes[0].set_yscale('log')
#plt.xscale('log')
axes[0].plot(range(maxIterations),mean_tv_errors_m,label = f'ULA h = {np.round(h,5)}')


errorTol = 0.1
h = (errorTol**2) / (2 * K * L)

tv_errors_samples = []

for samples in range(5):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_s = np.mean(np.array(tv_errors_samples),axis = 0)
axes[0].set_yscale('log')
#plt.xscale('log')
axes[0].plot(range(maxIterations),mean_tv_errors_s,label = f'ULA h = {np.round(h,5)}')

errorTol = 1.0
h = (errorTol**2) / (2 * K * L)

tv_errors_samples = []

for samples in range(5):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf))

    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_b = np.mean(np.array(tv_errors_samples),axis = 0)
axes[0].set_yscale('log')
#plt.xscale('log')
axes[0].plot(range(maxIterations),mean_tv_errors_b,label = f'ULA h = {np.round(h,5)}')

axes[0].legend()
axes[0].set_title("Discrete TV error Vs Iteration")
axes[0].set_ylabel("TV error")
axes[0].set_xlabel("Iteration")


T_max = 15
errorTol = 0.2
h = (errorTol**2) / (2 * K * L)
maxIterations = int(T_max / h)

tv_errors_samples = []

for samples in range(5):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf))
    
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)

axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'ULA h = {np.round(h,5)}')



errorTol = 0.1
h = (errorTol**2) / (2 * K * L)
maxIterations = int(T_max / h)

tv_errors_samples = []

for samples in range(5):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf))
    
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)

axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'ULA h = {np.round(h,5)}')


errorTol = 1.0
h = (errorTol**2) / (2 * K * L)
maxIterations = int(T_max / h)

tv_errors_samples = []

for samples in range(5):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations,gradf=gradf))
    
    
    tv_errors = []
    
    for i in range(maxIterations):
        hist_sample,_,_ = np.histogram2d(states[:i,0],states[:i,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors = np.mean(np.array(tv_errors_samples),axis = 0)

axes[1].set_yscale('log')
#plt.xscale('log')
axes[1].plot(np.linspace(0,T_max,num=maxIterations),mean_tv_errors,label = f'ULA h = {np.round(h,5)}')

#axes[1].legend()
axes[1].set_title("Discrete TV error Vs Time")
axes[1].set_ylabel("TV error")
axes[1].set_xlabel("Time")

fig.tight_layout()

plt.savefig("ULA_TVErrors")

#plt.savefig("ULA_no2 Tv error vs time")
plt.show()
'''
#%%

#Discrete TV error - Projected
'''
def discrete_tv_error(hist1,hist2):
    return np.sum(np.abs(hist1-hist2)) / hist1.shape[0]

empirical_size = 250000

bernoulli_samples = np.random.rand(empirical_size,1) < 0.5
standard_normal_samples = multivariate_normal.rvs(mean=np.zeros((2)),
                                                  size=empirical_size)
empirical_samples = ((standard_normal_samples - a)*bernoulli_samples + 
                    (standard_normal_samples + a)*(1-bernoulli_samples))

#%%
target_first_principal_axis = np.linalg.norm(empirical_samples @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(empirical_samples) @ np.array([1,1]))
target_second_principal_axis = np.linalg.norm(empirical_samples @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(empirical_samples) @ np.array([1,-1]))
#%%
target_first_hist,_ = np.histogram(target_first_principal_axis, bins=100, range=(-5,5), density=True)
target_second_hist,_ = np.histogram(target_second_principal_axis, bins=100, range=(-5,5), density=True)
#%%
np.sum(target_first_hist)
#%%

#TV Error - slopes matter
from scipy.stats import linregress

T_max = 1
h = 1e-3

slopes = {}

maxIterations_small = int(T_max / h)

fig,axes = plt.subplots()

tv_errors_samples = []


for samples in range(10):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations_small,gradf=gradf))
    
    
    tv_errors = []
    
    for i in range(maxIterations_small):
        hist_sample,_,_ = np.histogram2d(states[:i+1,0],states[:i+1,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_small = np.mean(np.array(tv_errors_samples),axis = 0)

axes.set_yscale('log')
#plt.xscale('log')
axes.plot(np.linspace(0,T_max,num=maxIterations_small),mean_tv_errors_small,label = f'ULA h = {np.round(h,5)}')

#slopes[h] = -(np.log(mean_tv_errors[0]) - np.log(mean_tv_errors[int(0.1/h)])) / 0.1

h = 0.01

maxIterations_big = int(T_max / h)

tv_errors_samples = []

for samples in range(10):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations_big,gradf=gradf))
    
    
    tv_errors = []
    
    for i in range(maxIterations_big):
        hist_sample,_,_ = np.histogram2d(states[:i+1,0],states[:i+1,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_big = np.mean(np.array(tv_errors_samples),axis = 0)

axes.set_yscale('log')
#plt.xscale('log')
axes.plot(np.linspace(0,T_max,num=maxIterations_big),mean_tv_errors_big,label = f'ULA h = {np.round(h,5)}')


#slopes[h] = -(np.log(mean_tv_errors[0]) - np.log(mean_tv_errors[int(0.1/h)])) / 0.1

h = 1e-4

maxIterations_med = int(T_max / h)


tv_errors_samples = []

for samples in range(10):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations_med,gradf=gradf))
    
    
    tv_errors = []
    
    for i in range(maxIterations_med):
        hist_sample,_,_ = np.histogram2d(states[:i+1,0],states[:i+1,1],bins=100,range = [[-5,5],[-5,5]],density=True)
        tv_errors.append(discrete_tv_error_2d(hist_sample,target_hist))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_med = np.mean(np.array(tv_errors_samples),axis = 0)

axes.set_yscale('log')
#plt.xscale('log')
axes.plot(np.linspace(0,T_max,num=maxIterations_med),mean_tv_errors_med,label = f'ULA h = {np.round(h,5)}')

#slopes[h] = -(np.log(mean_tv_errors[0]) - np.log(mean_tv_errors[int(0.1/h)])) / 0.1

axes.set_title("TV error vs time-ULA")
axes.set_xlabel("Time")
axes.set_ylabel("TV Error")

axes.legend()
plt.savefig("Error Vs Time - ULA-Non projected-hsmall")
plt.show()
#print(slopes)



#%%
slopes_sm_r = {}
slopes_sm_r[1e-3], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_med)[:25], np.log(mean_tv_errors_med)[:25])
slopes_sm_r[0.01], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_big)[:25], np.log(mean_tv_errors_big)[:25])
slopes_sm_r[1e-4], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_small)[:25], np.log(mean_tv_errors_small)[:25])
slopes_sm_r
#%%
slopes_sm = {}
slopes_sm[1e-3], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_med), np.log(mean_tv_errors_med))
slopes_sm[0.01], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_big), np.log(mean_tv_errors_big))
slopes_sm[1e-4], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_small), np.log(mean_tv_errors_small))
slopes_sm
#%%
plt.title('Pruned - 2D Error Vs time - ULA')
plt.yscale('log')
plt.plot(np.linspace(0,T_max,num=maxIterations_small)[:20], mean_tv_errors_small[:20],label='ULA - h=0.0001')
plt.yscale('log')
plt.plot(np.linspace(0,T_max,num=maxIterations_big)[:20], mean_tv_errors_big[:20],label = 'ULA - h=0.01')
plt.yscale('log')
plt.plot(np.linspace(0,T_max,num=maxIterations_med)[:20], mean_tv_errors_med[:20],label = 'ULA - h=0.0001')


plt.xlabel('Time')
plt.ylabel('TV error')

plt.savefig('Pruned - 2D Error Vs time - ULA')
plt.show()

#%%
#plt.yscale('log')
plt.plot(np.linspace(0,T_max,num=maxIterations_med)[1:400],np.log(mean_tv_errors_med)[1:400])
plt.plot(np.linspace(0,T_max,num=maxIterations_big)[1:160], np.log(mean_tv_errors_big)[1:160])
plt.plot(np.linspace(0,T_max,num=maxIterations_small)[1:], np.log(mean_tv_errors_small)[1:])
#%%
np.savetxt('mean_tv_errors_small.txt', mean_tv_errors_small, fmt='%f')
np.savetxt('mean_tv_errors_med.txt', mean_tv_errors_med, fmt='%f')
np.savetxt('mean_tv_errors_big.txt', mean_tv_errors_big, fmt='%f')
np.savetxt('h_values',np.array([1e-5,5e-5,2e-6]),fmt='%f')
np.savetxt('Iterations',np.array([maxIterations_med,maxIterations_small,maxIterations_big]),fmt='%f')
#%%
#Visually finding out when mixing happens
hs = np.linspace(0.01,0.1,5)
T_max = 300

for h in hs:
    maxIterations_1 = int(T_max / h)

    tv_errors_samples = []
    
    for samples in range(10):
        states = ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations_1,gradf=gradf)
        first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
        second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
        
        tv_errors = []
        
        for i in range(maxIterations_1):
            hist1,_ = np.histogram(first_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
            hist2,_ = np.histogram(second_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
            error1 = discrete_tv_error(hist1,target_first_hist)
            error2 = discrete_tv_error(hist2,target_second_hist)
            tv_errors.append((error1 + error2))
            
        tv_errors_samples.append(tv_errors)
    
    mean_tv_errors_1 = np.mean(np.array(tv_errors_samples),axis = 0)
    
    plt.yscale('log')
    plt.plot(np.linspace(0,T_max,num=maxIterations_1),mean_tv_errors_1,label = f'ULA h = {np.round(h,5)}') 

plt.title("Mixing time estimation")
plt.xlabel("Time")
plt.ylabel("Tv error")
plt.legend()
plt.savefig("Mixing time estimation")
plt.show()


#%%
maxIterations = int(T_max / h)

tv_errors_samples = []

for samples in range(10):
    states = ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations_1,gradf=gradf)
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
    second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
    
    tv_errors = []
    
    for i in range(maxIterations_1):
        hist1,_ = np.histogram(first_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
        hist2,_ = np.histogram(second_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
        error1 = discrete_tv_error(hist1,target_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append((error1 + error2))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_1 = np.mean(np.array(tv_errors_samples),axis = 0)

plt.yscale('log')
plt.plot(np.linspace(0,T_max,num=maxIterations_1),mean_tv_errors_1,label = f'ULA h = {h}')



#%%
#Testing discretization bias



'''
#%%
'''
sampled_states = np.array(states(step_size = h,
                         initial_distribution = multivariate_normal(cov= 1 * np.eye(2)),
                         max_iterations = maxIterations,
                         grad_potential=gradf))


            

'''
#%%
'''

#ULA - Measuring the discretization bias

Ts = [1000]
hs = np.linspace(0.01,0.1,10)

fig,axes = plt.subplots(1,1,figsize=(9,5))


for Tmax in Ts:
    mean_tv_errors = []
    
    for h in hs:
        
        maxIterations = int(Tmax/h)
        tv_errors_samples = []
        
        
        for samples in range(10):
            sampled_states = np.array(states(step_size = h,
                                     initial_distribution = multivariate_normal(cov= 1 * np.eye(2)),
                                     max_iterations = maxIterations,
                                     grad_potential=gradf))
            
            hist_sample,_,_ = np.histogram2d(sampled_states[:,0], sampled_states[:,1], bins=10,
                                             range = [[-5,5],[-5,5]],
                                             density=True)
            
            tv_errors_samples.append(discrete_tv_error_2d(hist_sample/100,
                                                          target_hist/100))
            
        #print(tv_errors_samples)
        mean_tv_errors.append(np.mean(np.array(tv_errors_samples),axis=0))
        
    axes.plot(hs,mean_tv_errors,'or',label=f'T={Tmax}',)
    
    m,b = np.polyfit(hs,mean_tv_errors,1)
    axes.plot(hs,m*hs + b,label=f'Regression Line for curve with T={Tmax}')


axes.set_xlabel('h')
axes.set_ylabel('Tv error')
axes.set_title("TV error Vs h at a fixed time T")
axes.grid(True)
axes.legend()

#plt.savefig('ULA - TV error vs h')
plt.show()



#%%
'''
'''
#hs = np.array([0.01,0.02,0.04,0.08])
hs = np.linspace(0.01,1,40)

fig,axes = plt.subplots(1,1,figsize=(9,5))


mean_tv_errors = []


Tmax = 1100
#for Tmax in Ts:
for h in hs:
    
    maxIterations = int(Tmax/h)
    tv_errors_samples = []
    
    
    for samples in range(1):
        sampled_states = np.array(states(step_size = h,initial_distribution =
                                         multivariate_normal(cov= 1/L * np.eye(2)),
                                         max_iterations = maxIterations, grad_potential=gradf))
        
        hist_sample,_,_ = np.histogram2d(sampled_states[:,0],sampled_states[:,1],bins=30,range = [[-4.5,4.5],[-4.5,4.5]],density=True)
        
        tv_errors_samples.append(discrete_tv_error_2d(hist_sample/100,target_hist/100))
        
        
    mean_tv_errors.append(np.mean(np.array(tv_errors_samples),axis=0))
    
axes.plot(hs,mean_tv_errors,'og',label=f'T={Tmax}')

m,b = np.polyfit(hs,mean_tv_errors,1)
axes.plot(hs,m*hs + b)
plt.show()


#%%
axes[1].plot(np.log(hs),np.log(mean_tv_errors),label=f'T={Tmax}')
m,b = np.polyfit(np.log(hs),np.log(mean_tv_errors),1)
print(f'slope for {Tmax} is {m}')
axes[1].plot(np.log(hs),m*np.log(hs)+b)





axes[0].set_title("ULA TV error at T vs h")
axes[0].set_xlabel("Stepsize h")
axes[0].set_ylabel("TV error")
axes[0].legend()

axes[1].set_title("ULA log(TV error) at T vs log(h)")
axes[1].set_xlabel("log(h)")
axes[1].set_ylabel("log(TV error)")

#plt.savefig('ULA Discretisation bias-Ts')
plt.show()

#%%

mean_tv_errors = []
fig,axes = plt.subplots(1,1,figsize=(9,5))
hs = [0.01,0.02,0.04,0.08]


Tmax = 40
#for Tmax in Ts:
for h in hs:
    
    maxIterations = int(Tmax/h)
    tv_errors_samples = []
    
    
    for samples in range(20):
        sampled_states = np.array(states(step_size = h,initial_distribution =
                                         multivariate_normal(cov= 1/L * np.eye(2)),
                                         max_iterations = maxIterations, grad_potential=gradf))
        
        first_principal_axis = np.linalg.norm(np.array(sampled_states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(sampled_states) @ np.array([1,1]))
        second_principal_axis = np.linalg.norm(np.array(sampled_states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(sampled_states) @ np.array([1,-1]))
        
        
        hist1,_ = np.histogram(first_principal_axis,bins=100,density=True,range=[-5,5])
        hist2,_ =np.histogram(second_principal_axis,bins=100,density=True,range=[-5,5])
        
        tv_errors_samples.append(discrete_tv_error(hist1,target_first_hist) + discrete_tv_error(hist2,target_second_hist))
        
    mean_tv_errors.append(np.mean(np.array(tv_errors_samples),axis=0))
    
axes.plot(hs,mean_tv_errors,'og',label=f'T={Tmax}')

m,b = np.polyfit(hs,mean_tv_errors,1)
#axes.plot(hs,m*hs + b)
#%%

mean_tv_errors = []
Tmax = 20
#for Tmax in Ts:
for h in hs:
    
    maxIterations = int(Tmax/h)
    tv_errors_samples = []
    
    
    for samples in range(20):
        sampled_states = np.array(states(step_size = h,initial_distribution =
                                         multivariate_normal(cov= 1/L * np.eye(2)),
                                         max_iterations = maxIterations, grad_potential=gradf))
        first_principal_axis = np.linalg.norm(np.array(sampled_states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(sampled_states) @ np.array([1,1]))
        second_principal_axis = np.linalg.norm(np.array(sampled_states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(sampled_states) @ np.array([1,-1]))
        
        
        hist1,_ = np.histogram(first_principal_axis,bins=100,density=True,range=[-5,5])
        hist2,_ =np.histogram(second_principal_axis,bins=100,density=True,range=[-5,5])
        
        tv_errors_samples.append(discrete_tv_error(hist1,target_first_hist) + discrete_tv_error(hist2,target_second_hist))
        
    mean_tv_errors.append(np.mean(np.array(tv_errors_samples),axis=0))
    
axes.plot(hs,mean_tv_errors,'or',label=f'T={Tmax}')

m,b = np.polyfit(hs,mean_tv_errors,1)
axes.plot(hs,m*hs + b)


mean_tv_errors = []
Tmax = 30
#for Tmax in Ts:
for h in hs:
    
    maxIterations = int(Tmax/h)
    tv_errors_samples = []
    
    
    for samples in range(20):
        sampled_states = np.array(states(step_size = h,initial_distribution =
                                         multivariate_normal(cov= 1/L * np.eye(2)),
                                         max_iterations = maxIterations, grad_potential=gradf))
        
        first_principal_axis = np.linalg.norm(np.array(sampled_states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(sampled_states) @ np.array([1,1]))
        second_principal_axis = np.linalg.norm(np.array(sampled_states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(sampled_states) @ np.array([1,-1]))
        
        
        hist1,_ = np.histogram(first_principal_axis,bins=100,density=True,range=[-5,5])
        hist2,_ = np.histogram(second_principal_axis,bins=100,density=True,range=[-5,5])
        
        tv_errors_samples.append(discrete_tv_error(hist1,target_first_hist) + discrete_tv_error(hist2,target_second_hist))
        
    mean_tv_errors.append(np.mean(np.array(tv_errors_samples),axis=0))
    
axes.plot(hs,mean_tv_errors,'ob',label=f'T={Tmax}')

m,b = np.polyfit(hs,mean_tv_errors,1)
axes.plot(hs,m*hs + b)



axes.set_title("ULA TV error at T vs h")
axes.set_xlabel("Stepsize h")
axes.set_ylabel("TV error")
axes.legend()

#plt.savefig('ULA Discretisation bias-proj tv error')
plt.show()




#%%
#plt.yscale('log')
#plt.xscale('log')
plt.plot(np.log(hs),np.log(mean_tv_errors),label=f'T={Tmax}')

m,b = np.polyfit(np.log(hs),np.log(mean_tv_errors),1)
plt.plot(np.log(hs),m*np.log(hs)+b)
#plt.plot(hs,m*hs + b)
print(m,b)
#%%

plt.plot(hs,mean_tv_errors)

#%%
#slope, intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations), np.log10(mean_tv_errors))
(np.log(mean_tv_errors[100]) - np.log(mean_tv_errors[10]) ) / (90*h) 
#print(slope)
'''
#%%
#TV error VS iterations 
'''

maxIterations = 700

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
        error1 = discrete_tv_error(hist1,_first_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append((error1 + error2)/2)
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_mid = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')
#plt.xscale('log')
plt.plot(range(maxIterations),mean_tv_errors_mid,label = f'ULA h = {h}')


errorTol = 0.1
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

mean_tv_errors_small = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')
#plt.xscale('log')
plt.plot(range(maxIterations),mean_tv_errors_small,label = f'ULA h = {h}')

errorTol = 1.0
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

mean_tv_errors_large = np.mean(np.array(tv_errors_samples),axis = 0)
plt.yscale('log')
#plt.xscale('log')
plt.plot(range(maxIterations),mean_tv_errors_large,label = f'ULA h = {h}')

plt.legend()
plt.title("Discrete TV error Vs Iteration")
plt.ylabel("TV error")
plt.xlabel("Iteration")
plt.savefig("ULA_TVError_Iteration")
plt.show()
#%%
slopesy = {}
slopesy[0.01], intercept, r_value, p_value, std_err = linregress(range(1,400), np.log(mean_tv_errors_mid)[1:400])
slopesy[0.0025], intercept, r_value, p_value, std_err = linregress(range(1,400), np.log(mean_tv_errors_small)[1:400])
slopesy[0.25], intercept, r_value, p_value, std_err = linregress(range(1,400), np.log(mean_tv_errors_large)[1:400])
slopesy
#%%

#TV error Vs Time

T_max = .1

h = 0.0001

maxIterations_1 = int(T_max / h)

tv_errors_samples = []

for samples in range(10):
    states = ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations_1,gradf=gradf)
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
    second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
    
    tv_errors = []
    
    for i in range(maxIterations_1):
        hist1,_ = np.histogram(first_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
        hist2,_ = np.histogram(second_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
        error1 = discrete_tv_error(hist1,target_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append((error1 + error2))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_1 = np.mean(np.array(tv_errors_samples),axis = 0)

plt.yscale('log')
plt.plot(np.linspace(0,T_max,num=maxIterations_1),mean_tv_errors_1,label = f'ULA h = {h}')

h = 0.01
maxIterations_3 = int(T_max / h)
tv_errors_samples = []

for samples in range(10):
    states = ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations_3,gradf=gradf)
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
    second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
    
    tv_errors = []
    
    for i in range(maxIterations_3):
        hist1,_ = np.histogram(first_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
        hist2,_ = np.histogram(second_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
        error1 = discrete_tv_error(hist1,target_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append((error1 + error2))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_3 = np.mean(np.array(tv_errors_samples),axis = 0)

plt.yscale('log')
plt.plot(np.linspace(0,T_max,num=maxIterations_3),mean_tv_errors_3,label = f'ULA h = {h}')

h = 0.001

tv_errors_samples = []
maxIterations_2 = int(T_max / h)

for samples in range(10):
    states = ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations = maxIterations_2,gradf=gradf)
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
    second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
    
    tv_errors = []
    
    for i in range(maxIterations_2):
        hist1,_ = np.histogram(first_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
        hist2,_ = np.histogram(second_principal_axis[:i+1],bins=100,range=(-5.2,5.2),density=True)
        error1 = discrete_tv_error(hist1,target_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append((error1 + error2))
        
    tv_errors_samples.append(tv_errors)

mean_tv_errors_2 = np.mean(np.array(tv_errors_samples),axis = 0)


plt.yscale('log')
plt.plot(np.linspace(0,T_max,num=maxIterations_2),mean_tv_errors_2,label = f'ULA h = {h}')


plt.legend()
plt.title("Projected discrete TV error Vs Time")
plt.xlabel("Time")
plt.ylabel("TV error")
plt.yticks(np.arange(0.3,0.4,0.01))
plt.grid(axis='y', linestyle='-')
plt.savefig("ULA_projected_TVError_Time_hsmall")

#plt.savefig("ULA_no2 Tv error vs time")
plt.show()
#%%
slopes_smp_r = {}
slopes_smp_r[0.001], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_2)[:20], np.log(mean_tv_errors_2)[:20])
slopes_smp_r[0.0001], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_1)[:100], np.log(mean_tv_errors_1)[:100])
slopes_smp_r[0.01], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_3)[:4], np.log(mean_tv_errors_3)[:4])
slopes_smp_r
#%%
slopes_smp = {}
slopes_smp[0.001], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_2), np.log(mean_tv_errors_2))
slopes_smp[0.0001], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_1), np.log(mean_tv_errors_1))
slopes_smp[0.01], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_3), np.log(mean_tv_errors_3))
slopes_smp

#%%
slopes_sm_r = {}
slopes_sm_r[0.01], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_2)[:20], np.log(mean_tv_errors_2)[:20])
slopes_sm_r[0.002], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_1)[:100], np.log(mean_tv_errors_1)[:100])
slopes_sm_r[0.05], intercept, r_value, p_value, std_err = linregress(np.linspace(0,T_max,num=maxIterations_3)[:4], np.log(mean_tv_errors_3)[:4])
slopes_sm_r

#%%
'''
'''
import bisect

def ULAMixing(h,initial_distribution,gradf,
              quantile, target_quantile, mixing_dimension = 0 ,
              maxIterations = 10000, errorTol = 0.2):
    
    x = initial_distribution.rvs()
    root_two_h = np.sqrt(2*h)
    dim = x.shape[0]

    
    sorted_mixing_states = [x[mixing_dimension]]
    
    for i in range(maxIterations):
        
        x = x + root_two_h * np.random.randn(dim) - h * gradf(x)
        
        bisect.insort(sorted_mixing_states,x[mixing_dimension])
        
        #len(sorted_mixing_states) = i+2
        if (i+2) % 4 == 0:
            empirical_quantile = sorted_mixing_states[int(quantile * (i+2)) - 1]
        else:
            empirical_quantile = sorted_mixing_states[int(quantile * (i+2))]
        
        #sorted_mixing_states.append(x[mixing_dimension])
        #empirical_quantile = np.quantile(sorted_mixing_states ,quantile)
            
            
        if np.abs(empirical_quantile - target_quantile) < errorTol:
            return i+1
    
    return i+1

#%%
def ULA_quantile_error(h,initial_distribution,gradf,
              quantile, target_quantile, mixing_dimension = 0 ,
              maxIterations = 10000, errorTol = 0.2):
    
    x = initial_distribution.rvs()
    root_two_h = np.sqrt(2*h)
    dim = x.shape[0]
    quantile_errors = []

    
    sorted_mixing_states = [x[mixing_dimension]]
    
    for i in range(maxIterations):
        
        x = x + root_two_h * np.random.randn(dim) - h * gradf(x)
        
        bisect.insort(sorted_mixing_states,x[mixing_dimension])
        
        #len(sorted_mixing_states) = i+2
        if (i+2) % 4 == 0:
            empirical_quantile = sorted_mixing_states[int(quantile * (i+2)) - 1]
        else:
            empirical_quantile = sorted_mixing_states[int(quantile * (i+2))]
        
        #sorted_mixing_states.append(x[mixing_dimension])
        #empirical_quantile = np.quantile(sorted_mixing_states ,quantile)
            
            
        quantile_errors.append(np.abs(empirical_quantile - target_quantile))
        
    return quantile_errors
#%%
'''
'''
d = 5
ULA_quantile_error(h,multivariate_normal(cov= 1/L * np.eye(d)),gradf,
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 10000)

#%%
    
a = np.arange(5)
print(a)
l = len(a)
if l%4 == 0:
    print(a[int(0.75*l)-1])
else:
    print(a[int(0.75*l)])

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
dimensions = np.arange(5,16)


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


#%%
plt.yscale('log')
plt.xscale('log')
plt.plot(dimensions,mixing_times,'og')
plt.title('ULA tmix 0.75 quantile')
#plt.savefig('ULA tmix vs dim-20runs') 
plt.show()
#%%
m,b = np.polyfit(dimensions,mixing_times,1)
plt.title('ULA tmix 0.75 quantile - tmix vs dim')
plt.plot(dimensions,m*dimensions + b)
plt.plot(dimensions,mixing_times,'og')
plt.xlabel('Dimension')
plt.ylabel('Tmix')
plt.savefig('ULA tmix vs dim - linear')
plt.show()
#%%
slope, intercept, r_value, p_value, std_err = linregress(np.log(dimensions), np.log(mixing_times))
slope
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
plt.title('ULA tmix 0.75 quantile - tmix vs dim')
plt.xlabel('Dimension')
plt.ylabel('Tmix')
plt.plot(newX, myExpFunc(newX, *popt), 'b')
plt.savefig('ULA tmix vs dim')
plt.show()
#%%
popt

#%%
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(np.log(dimensions), np.log(mixing_times))


xfid = np.linspace(2,14)     # This is just a set of x to plot the straight line 

plt.plot(dimensions, mixing_times, 'k.')
plt.plot(xfid, xfid*slope+intercept)
plt.grid()
plt.show()

slope
#%%


def ULAl2Mixing(h,initial_distribution,gradf,
              mixing_dimension = 0 , target_l2norm = 2,
              maxIterations = 10000, errorTol = 0.2):
    
    x = initial_distribution.rvs()
    root_two_h = np.sqrt(2*h)
    dim = x.shape[0]
    
    
    empirical_avgl2norm = x[mixing_dimension]**2
    
    for i in range(maxIterations):
        
        x = x + root_two_h * np.random.randn(dim) - h * gradf(x)
        
        empirical_avgl2norm = (empirical_avgl2norm * (i+1) + x[mixing_dimension]**2) / (i+2)
                    
        
        if np.abs(target_l2norm - np.sqrt(empirical_avgl2norm)) < errorTol:
            return i+1
    
    return i+1
#%%

mixing_times = []
dimensions = np.arange(5,16)


for d in dimensions:
    h = (errorTol**2) / (d * K * L)
    
    #diagonal_entries = np.flip(np.linspace(1.0,4.0,num=d))
    diagonal_entries = np.array([4.0] + list(np.ones(d-1)))

    gradf = lambda x : np.diag(1 / diagonal_entries) @ x
    
    mixing_time_sum = 0
    
    for i in range(200):
        
        mixing_time_sum += ULAl2Mixing(h,multivariate_normal(cov= 1/L * np.eye(d)),gradf,
                                       target_l2norm = 2,
                                      errorTol=errorTol,maxIterations = 10000)
        
    mixing_time_avg = mixing_time_sum / 200
    
    mixing_times.append(mixing_time_avg)

#%%
plt.yscale('log')
plt.xscale('log')
plt.plot(dimensions,mixing_times,'og')
plt.title('ULA l2 mixing time')
#plt.savefig('ULA l2 mix')  
plt.show()
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
plt.title('ULA l2 mixing time')
plt.xlabel('dimensions')
plt.ylabel('mixing time')
plt.savefig('ULA l2 mix 200runs')

#%%
popt


#%%

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(dimensions, np.log10(mixing_times))


xfid = np.linspace(0.7,1.3)     # This is just a set of x to plot the straight line 

plt.plot(np.log10(dimensions), np.log10(mixing_times), 'k.')
plt.plot(xfid, xfid*slope+intercept)
plt.show()

slope
#%%

#Histogram plot
d = 10
errorTol = 0.2

diagonal_entries = np.array([4.0] + list(np.ones(d-1)))
gradf = lambda x : np.diag(1/diagonal_entries) @ x


#Constants bounding the rate of growth of the Hessian, L is upper and m is lower bound
L  = 1
m = 0.25
K = L/m

diagonal_entries
#%%

#Histogram Test
h = (errorTol**2) / (d * K * L)
xs = []

#T_max = 120
noOfSampledPoints = 20000
maxIterations = 4300
for i in range(noOfSampledPoints):
    final_state = ULA(h,multivariate_normal(cov= 1/L * np.eye(d)) ,maxIterations = maxIterations, gradf = gradf)
    xs.append(final_state[0])
#%%
x_points = np.linspace(-5,5,100)
target_curve = scipy.stats.norm.pdf(x_points,scale=2)
#%%

empirical_xs = scipy.stats.norm.rvs(scale=2**0.5,size=250000)

#%%
plt.title(f"ULA - {noOfSampledPoints} points, {maxIterations} iterations, h = {h}")
_,_,_ = plt.hist(xs,bins=100,range=(-5,5),density=True,alpha=0.6,label='ULA')
plt.plot(x_points,target_curve,label='target pdf')
#_,_,_ = plt.hist(empirical_xs,bins=100,range=(-5,5),density=True,alpha=0.6,label='target')
plt.legend()
plt.title(f"ULA_histogram")
plt.savefig(f"ULA_histogram_var4")
plt.show()


#%%
#Tmix vs 1/delta
d = 5
diagonal_entries = np.array([4.0] + list(np.ones(d-1)))
gradf = lambda x : np.diag(1 / diagonal_entries) @ x
mixing_times = []

for invErrorTol in range(2,10):
    errorTol = 1 / invErrorTol
    h = (errorTol**2) / (d * K * L)
    
    mixing_time_sum = 0
    
    for i in range(100):
        
        mixing_time_sum += ULAMixing(h,multivariate_normal(cov= 1/L * np.eye(d)),gradf,
                                      quantile,target_quantile,
                                      errorTol=errorTol,maxIterations = 10000)
        
    mixing_time_avg = mixing_time_sum / 100
    
    mixing_times.append(mixing_time_avg)

#%%
plt.xscale('log')
plt.yscale('log')
plt.plot(range(2,10),mixing_times,'og')
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
plt.savefig('tmix  vs delta, d = 15, 200runs')

#%%
popt


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
#%%

f = lambda theta : np.squeeze(-y.T @ X @ theta + alpha * theta.T @ sigma_x @ theta + np.sum(np.log(1 + np.exp(X @ theta))))
#gradf = lambda theta : -X.T @ y + alpha * sigma_x @ theta + (np.sum(X/(1 + np.exp(-X @ theta)),axis=0).reshape((-1,1)))

gradf_constant = -X.T @ y 

def gradf(theta):
    theta_column = theta.reshape((-1,1))
    return np.squeeze(gradf_constant + alpha * sigma_x @ theta_column + (np.sum(X/(1 + np.exp(-X @ theta_column)),axis=0).reshape((-1,1))))

#%%
eigvals = np.linalg.eigvalsh(sigma_x)
L = (0.25 * n + alpha) * eigvals[-1]
m = alpha * eigvals[0]

#%%
maxIterations = 4000
errorTol = 0.2
h = (errorTol**2) / (d * K * L)

l1_errors_samples = []

for samples in range(50):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations ,gradf))
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/d
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'ULA h = {np.round(h,5)}')


errorTol = 0.1
h = (errorTol**2) / (d * K * L)

l1_errors_samples = []

for samples in range(50):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations ,gradf))
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/d
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'ULA h = {np.round(h,5)}')

errorTol = 1.0
h = (errorTol**2) / (d * K * L)

l1_errors_samples = []

for samples in range(50):
    states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations ,gradf))
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/d
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'ULA h = {np.round(h,5)}')

plt.legend()
plt.title('ULA - Mean l1 error- Logistic regression')
plt.ylabel('l1 Error')
plt.xlabel('Iteration')
plt.savefig('ULA-Mean l1 error, no pre conditioning')


#%%
#Preconditioning

eigvals,eigvecs = np.linalg.eigh(sigma_x)
D = np.diag(eigvals)
sigma_x_root = eigvecs @ np.sqrt(D) @ eigvecs.T
sigma_x_inv_root = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T


f_preconditoned = lambda theta : f(sigma_x_inv_root @ theta.reshape((-1,1)))
#gradf = lambda theta : -X.T @ y + alpha * sigma_x @ theta + (np.sum(X/(1 + np.exp(-X @ theta)),axis=0).reshape((-1,1)))



def gradf_preconditioned(theta):
    v = gradf(sigma_x_inv_root @ theta.reshape((-1,1))).reshape((-1,1))
    return np.squeeze(sigma_x_inv_root @ v)


L = (0.25 * n + alpha) 
m = alpha


#%%
maxIterations = 4000
errorTol = 0.2
h = (errorTol**2) / (d * K * L)

l1_errors_samples = []

for samples in range(50):
    preconditioned_states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations ,gradf_preconditioned))
    states = preconditioned_states @ sigma_x_root
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/d
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'ULA ')


errorTol = 0.1
h = (errorTol**2) / (d * K * L)

l1_errors_samples = []

for samples in range(50):
    preconditioned_states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations ,gradf_preconditioned))
    states = preconditioned_states @ sigma_x_root
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/d
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'ULA Small')

errorTol = 1.0
h = (errorTol**2) / (d * K * L)

l1_errors_samples = []

for samples in range(50):
    preconditioned_states = np.array(ULA_states(h,multivariate_normal(cov= 1/L * np.eye(2)),maxIterations ,gradf_preconditioned))
    states = preconditioned_states @ sigma_x_root
    rolling_means = np.cumsum(states,axis=0) / np.arange(1,maxIterations+2).reshape((maxIterations+1,1))
    l1_errors = np.sum(np.abs(rolling_means-theta_true.T),axis=1)
    
    l1_errors_samples.append(l1_errors)

mean_l1_errors = np.mean(np.array(l1_errors_samples),axis = 0)/d
plt.yscale('log')
plt.plot(range(maxIterations+1),mean_l1_errors,label = f'ULA Large')

plt.legend()
plt.title('ULA - Mean l1 error- Logistic regression')
plt.ylabel('l1 Error')
plt.xlabel('Iteration')
plt.savefig('ULA-Mean l1 error, pre conditioning')
#%%
'''
