# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:02:16 2021

@author: deepa
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import odeint

import bisect

NAME = 'HMC'
#%%

def update(position, velocity, grad_position, grad_potential,
           leapfrog_steps, step_size):
    #returns position and grad_position
    for i in range(leapfrog_steps):
        
        position += step_size * velocity - ((step_size)**2)/2 * grad_position
        grad_position_new = grad_potential(position)
        velocity += -step_size/2 * (grad_position + grad_position_new)
        grad_position = grad_position_new
    
    return position



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
    leapfrog_steps = int(flow_time/step_size)
    position = initial_distribution.rvs()
    dimension = position.shape[0]
    
    for j in range(max_iterations):
        
        velocity = np.random.randn(dimension)
        
        #We store the value of grad to reduce the number of computations
        grad_position = grad_potential(position)
        
        for i in range(leapfrog_steps):
            
            position += step_size * velocity - ((step_size)**2)/2 * grad_position
            grad_position_new = grad_potential(position)
            velocity += -step_size/2 * (grad_position + grad_position_new)
            grad_position = grad_position_new
    
    return position
#%%
def set_step_size(error_tol, dimension,
                 condition_number, upper_hessian_bound, lower_hessian_bound):
    
    return np.power(dimension, -0.25) * np.sqrt(error_tol) 
#%%

def states(initial_distribution, step_size, max_iterations = 10000,
           grad_potential = None, potential = None, target_density = None,
           flow_time=1.0):

    leapfrog_steps = int(flow_time/step_size)
    
    position = initial_distribution.rvs()
    positions = [position]
    
    dimension = position.shape[0]
    
    for j in range(max_iterations):
        
        velocity = np.random.randn(dimension)
        
        #We store the value of grad to reduce the number of computations
        grad_position = grad_potential(position)
        
        for i in range(leapfrog_steps):
            
            position += step_size * velocity - ((step_size)**2)/2 * grad_position
            grad_position_new = grad_potential(position)
            velocity += -step_size/2 * (grad_position + grad_position_new)
            grad_position = grad_position_new
        
        positions.append(position)
    
    return np.array(positions)

#%%
    

def three_fourth_quantile_mixing(step_size,initial_distribution,
                                 potential, grad_potential, target_density,
                                 target_quantile, mixing_dimension = 0,
                                 max_iterations = 10000, error_tol = 0.2,
                                 flow_time = 1.0):

    
    leapfrog_steps = int(flow_time/step_size)
    position = initial_distribution.rvs()
    dimension = position.shape[0]
    
    sorted_states = [position[mixing_dimension]]
    
    for i in range(max_iterations):
        
        velocity = np.random.randn(dimension)
        
        #We store the value of grad to reduce the number of computations
        grad_position = grad_potential(position)
        
        for step in range(leapfrog_steps):
            
            position += step_size * velocity - ((step_size)**2)/2 * grad_position
            grad_position_new = grad_potential(position)
            velocity += -step_size/2 * (grad_position + grad_position_new)
            grad_position = grad_position_new
        
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