# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 07:21:28 2021

@author: deepa
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import odeint

import bisect

NAME = 'exactHMC'

#%%
C_inv = np.diag([0.5] + list(np.ones(4)))
grad_potential = lambda x : C_inv @ x

#%%
def final_state(initial_distribution, step_size, max_iterations = 10000,
                grad_potential = None, potential = None, target_density = None,
                flow_time=1.0):
    
    """
    Hamiltonian Monte Carlo
    
    Hyperparmeters :
    
    Algo:
        1) Draw X0 ~ initial_distribution. X = X0
        2) Pick V0 ~ N(0,Id)
        3) X = X + hV0 -
    
    """
    position = initial_distribution.rvs()
    print(position)
    dimension = position.shape[0]
    C = np.diag([2.0] + list(np.ones(dimension - 1)))
    print(C)
    rootC = C**(0.5)
    print(rootC)
    rootC_inv = np.diag([1/np.sqrt(2)] + list(np.ones(dimension - 1)))
    print(rootC_inv)
    
    for j in range(max_iterations):
        
        velocity = np.random.randn(dimension)
        position = np.cos(flow_time * rootC_inv)@position + rootC*np.sin(flow_time * rootC_inv)@velocity
        print(position)
    
    return position
#%%
print(final_state(multivariate_normal(cov=np.eye(5), allow_singular = True), 
                  0.01, max_iterations = 10, flow_time=1.0, grad_potential = grad_potential))

#%%
def set_step_size(error_tol, dimension,
                 condition_number, upper_hessian_bound, lower_hessian_bound):
    
    return 0
#%%

def states(initial_distribution, step_size, max_iterations = 10000,
           grad_potential = None, potential = None, target_density = None,
           flow_time=1.0):

    position = initial_distribution.rvs()
    dimension = position.shape[0]
    C = np.diag([2.0] + list(np.ones(dimension - 1)))
    rootC = C**(0.5)
    rootC_inv = np.diag([1/2.0] + list(np.ones(dimension - 1)))
    
    positions = [position]
    
    for j in range(max_iterations):

        velocity = np.random.randn(dimension)
        position = np.cos(flow_time * rootC_inv)*position + rootC*np.sin(flow_time * rootC_inv)*velocity
    
    return np.array(positions)

#%%
    

def three_fourth_quantile_mixing(step_size,initial_distribution,
                                 potential, grad_potential, target_density,
                                 target_quantile, mixing_dimension = 0,
                                 maxIterations = 10000, error_tol = 0.2,
                                 flow_time = 1.0):

    
    position = initial_distribution.rvs()
    dimension = position.shape[0]
    C = np.diag([2.0] + list(np.ones(dimension - 1)))
    rootC = C**(0.5)
    rootC_inv = np.diag([1/2.0] + list(np.ones(dimension - 1)))
    
    sorted_states = [position[mixing_dimension]]
    
    for i in range(maxIterations):
        
        velocity = np.random.randn(dimension)
        position = np.cos(flow_time * rootC_inv)@position + rootC*np.sin(flow_time * rootC_inv)@velocity
        
        bisect.insort(sorted_states,position[mixing_dimension])
        
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