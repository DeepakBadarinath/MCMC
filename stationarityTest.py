# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:21:04 2021

@author: deepa
"""


import numpy as np
import matplotlib.pyplot as plt


from MCMC_methods import *


#%%

""" 
The target density is a symmetric mixture of a pair of 2D gaussians, with means on (0.5,0.5) and -(0.5,0.5)

Axes of maximal variation is the x=y line and x=-y line
We look at the distribution projected to these axes

"""
a = np.array([0.5,0.5])

target_density = lambda x : 0.5 * (multivariate_normal.pdf(x,mean=a) + multivariate_normal.pdf(x,mean=-a))

# target density = exp(-f(x)), gradf is the gradient of this f
gradf = lambda x : x - a + 2 * a * 1/(1 + 2 * np.exp(np.dot(x,a)))

#Constants bounding the rate of growth of the Hessian, L is upper and m is lower bound
L  = 1
m = 0.5
K = L/m
#%%
"""
MRW stationary distribution on principal axis

"""
#StepSize taken as per DwivediChenWainrightYu2019 4.1

h = 1 / (2 * K * L)

ratios = []

plt.title("Convergence to stationarity on principal axis - MRW") 
plt.xlabel('Iteration')
plt.ylabel('State')

for i in range(10):

    _,states,ratio = MRW(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000)
    
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

    plt.plot(range(1001),first_principal_axis,alpha=0.3,color='r')
    
    ratios.append(ratio)


#plt.plot()
print(f"MRW acceptance ratio is {sum(ratios) / len(ratios)}")

#plt.savefig("Stationarity-MRW")
plt.show()
#%%
empirical_size = 250000
z = np.random.rand(empirical_size,1) < 0.5

empirical_samples = multivariate_normal.rvs(mean=a,size=empirical_size) * z + (1-z) * multivariate_normal.rvs(mean=-a,size=empirical_size)
empirical_xs = empirical_samples[:,0]
empirical_ys = empirical_samples[:,1]

_,_,_ = plt.hist(empirical_xs,bins=100,range=(-5,5),density=True)
#plt.show()
_,_,_ = plt.hist(empirical_ys,bins=100,range=(-5,5),density=True)
plt.show()



#%%
xs = []
ys = []

for i in range(1000):
    final_state,_,_ = MRW(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000)
    xs.append(final_state[0])
    ys.append(final_state[1])
#%%
_,_,_ = plt.hist(xs,bins=100,range=(-5,5),density=True)
_,_,_ = plt.hist(empirical_xs,bins=100,range=(-5,5),density=True)
plt.show()

_,_,_ = plt.hist(empirical_xs,bins=100,range=(-5,5),density=True)
_,_,_ = plt.hist(xs,bins=100,range=(-5,5),density=True)
plt.show()

#%%
"""
ULA stationary distribution on principal components 

"""


errorTol = 0.5

h = (errorTol**2) / (2 * K * L)



plt.title(f"Convergence to stationarity on principal axis - ULA, Tol = {errorTol}") 
plt.xlabel('Iteration')
plt.ylabel('State')

for i in range(10):

    _,states = ULA(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000, gradf = gradf)
    
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

    plt.plot(range(1001),first_principal_axis,alpha=0.3,color='g')
    


plt.savefig("Stationarity-ULA")
plt.show()
#%%
"""
MALA stationary distribution on principal componenents

"""

h = 1 / L * (min(1 / np.sqrt(2 * K), 1 / 2))
ratios = []

plt.title("Convergence to stationarity on principal axis - MALA") 
plt.xlabel('Iteration')
plt.ylabel('State')

for i in range(10):

    _,states,ratio = MALA(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000, gradf = gradf)
    
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

    plt.plot(range(1001),first_principal_axis,alpha=0.3,color='b')
    
    ratios.append(ratio)

print(f"MALA acceptance ratio is {sum(ratios) / len(ratios)}")

plt.savefig("Stationarity-MALA")
plt.show()
#%%
"""
Discrete TV Error Vs Iterations
"""

def discrete_tv_error(hist1,hist2):
    return np.sum(np.abs(hist1-hist2))


empirical_size = 250000
z = np.random.rand(empirical_size,1) < 0.5
empirical_samples = multivariate_normal.rvs(mean=a,size=empirical_size) * z + (1-z) * multivariate_normal.rvs(mean=-a,size=empirical_size)
target_first_principal_axis = np.linalg.norm(np.array(empirical_samples) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(empirical_samples) @ np.array([1,1]))
target_second_principal_axis = np.linalg.norm(np.array(empirical_samples) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(empirical_samples) @ np.array([1,-1]))

target_first_hist,_ = np.histogram(target_first_principal_axis, bins=100, range=(-5.2,5.2), density=True)
target_second_hist,_ = np.histogram(target_second_principal_axis, bins=100, range=(-5.2,5.2), density=True)
#%%

#MALA
h = 1 / L * (min(1 / np.sqrt(2 * K), 1 / 2))

_,states,_ = MALA(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000, gradf = gradf)

first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))

tv_errors = []

for i in range(1000):
    hist1,_ = np.histogram(first_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
    hist2,_ = np.histogram(second_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
    error1 = discrete_tv_error(hist1,target_first_hist)
    error2 = discrete_tv_error(hist2,target_second_hist)
    tv_errors.append(error1 + error2)
 
plt.plot(range(1000),tv_errors,label = 'MALA')


#MRW

h = 1 / (2 * K * L)
_,states,_ = MRW(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000)

first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))

tv_errors = []

for i in range(1000):
    hist1,_ = np.histogram(first_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
    hist2,_ = np.histogram(second_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
    error1 = discrete_tv_error(hist1,target_first_hist)
    error2 = discrete_tv_error(hist2,target_second_hist)
    tv_errors.append(error1 + error2)

 
plt.plot(range(1000),tv_errors, label = 'MRW')


#ULA


errorTol = 1.0
h = (errorTol**2) / (2 * K * L)
_,states = ULA(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000, gradf = gradf)

first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))

tv_errors = []

for i in range(1000):
    hist1,_ = np.histogram(first_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
    hist2,_ = np.histogram(second_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
    error1 = discrete_tv_error(hist1,target_first_hist)
    error2 = discrete_tv_error(hist2,target_second_hist)
    tv_errors.append(error1 + error2)
 
plt.plot(range(1000),tv_errors,label = 'ULA')

plt.title("Discrete TV errors Vs iteration")
plt.xlabel("Iteration")
plt.ylabel("Discrete TV Error")
plt.legend()

plt.savefig("Discrete TV errors Vs iteration")
plt.show()
#%%
"""
ULA Discrete TV error VS StepSize

"""
errorTols = [0.5,1.0,1.5]
graph_names = ["ULA small", "ULA moderate", "ULA large"]

for no,errorTol in enumerate(errorTols):
    h = (errorTol**2) / (2 * K * L)
    _,states = ULA(target_density,h,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000, gradf = gradf)
    
    first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))
    second_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,-0.5],[-0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,-1]))
    
    tv_errors = []
    
    for i in range(1000):
        hist1,_ = np.histogram(first_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
        hist2,_ = np.histogram(second_principal_axis[:i],bins=100,range=(-5.2,5.2),density=True)
        error1 = discrete_tv_error(hist1,target_first_hist)
        error2 = discrete_tv_error(hist2,target_second_hist)
        tv_errors.append(error1 + error2)
     
    plt.plot(range(1000),tv_errors,label = graph_names[no])
    
plt.title("ULA Stepsizes - TV error VS Iteration")
plt.xlabel("Iteration")
plt.ylabel("Discrete TV Error")
plt.legend()

plt.savefig("ULA Stepsizes - TV error VS Iteration")
plt.show()

#%%
"""
Autocorrelation Plots

"""


from statsmodels.graphics.tsaplots import plot_acf


h_MRW = 1 / (2 * K * L)
errorTol = 0.5
h_ULA = (errorTol**2) / (2 * K * L)
h_MALA = 1 / L * (min(1 / np.sqrt(2 * K), 1 / 2))


fig,axes = plt.subplots()
axes.set_title("Autocorrelation Plots for various MCMC methods")
axes.set_xlabel("Lag")
axes.set_ylabel("Autocorrelation")

_,states,_ = MALA(target_density,h_MALA,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000, gradf = gradf)
first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

kwargs = {'label':'MALA','alpha': 0.2,'color':'b','lw':1}
plot_acf(first_principal_axis, ax = axes ,**kwargs)
axes.legend()

_,states,_ = MRW(target_density,h_MRW,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000)
first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

kwargs = {'label':'MRW','alpha': 0.2,'color':'r','lw':1}
plot_acf(first_principal_axis, ax = axes , **kwargs)

_,states = ULA(target_density,h_ULA,multivariate_normal(cov= 1/L * np.eye(2)) ,maxIterations = 1000,gradf=gradf)
first_principal_axis = np.linalg.norm(np.array(states) @ np.array([[0.5,0.5],[0.5,0.5]]),axis = -1) * np.sign(np.array(states) @ np.array([1,1]))

kwargs = {'label':'ULA','alpha': 0.2,'color':'m','lw':1}
plot_acf(first_principal_axis, ax = axes , **kwargs)

axes.legend()
plt.savefig("Autocorrelation of different MCMC methods")
plt.show()