# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:02:33 2021

@author: deepa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import odeint

import bisect

NAME = 'mHMC'
#%%

def update(position, velocity, grad_position,
           grad_potential, potential,
           leapfrog_steps, step_size,
           acceptances):
    
    #Store old data
    old_position = position.copy()
    old_grad_position = grad_position.copy()
    old_velocity = velocity.copy()
    
    #Perform velocity verlet
    for i in range(leapfrog_steps):
        
        position += step_size * velocity - ((step_size)**2)/2 * grad_position
        grad_position_new = grad_potential(position)
        velocity += -step_size/2 * (grad_position + grad_position_new)
        grad_position = grad_position_new
    
    new_hamiltonian = potential(position) + 0.5 * np.sum(velocity*velocity)
    old_hamiltonian = potential(old_position) + 0.5 * np.sum(old_velocity*old_velocity)
    
    if new_hamiltonian <= old_hamiltonian:
        acceptances += 1
        return position, grad_position,acceptances
    
    acceptance_probability = np.exp(old_hamiltonian - new_hamiltonian)
    if np.random.uniform() < acceptance_probability:
        acceptances += 1
        return position, grad_position,acceptances
    
    #Reject 
    return old_position, old_grad_position, acceptances



#%%
def final_state(initial_distribution, step_size, max_iterations = 10000,
                grad_potential = None, potential = None, target_density = None,
                flow_time=1.0, leapfrog_steps = None,
                show_acceptance_ratio = False):
    
    """
    Hamiltonian Monte Carlo
    
    Hyperparmeters :
    
    Algo:
        1) Draw X0 ~ initial_distribution. X = X0
        2) Pick V0 ~ N(0,Id)
        3) X = X + hV0 -
    
    """
    if leapfrog_steps == None:
        leapfrog_steps = max(int(flow_time/step_size),1)
    position = initial_distribution.rvs()
    dimension = position.shape[0]
    acceptances = 0
    
    for j in range(max_iterations):
        
        velocity = np.random.randn(dimension)
        
        #We store the value of grad to reduce the number of computations
        grad_position = grad_potential(position)
        
        position, grad_position, acceptances = update(position, velocity, grad_position,
                                                      grad_potential, potential,
                                                      leapfrog_steps, step_size,
                                                      acceptances)
    acceptance_ratio = acceptances/max_iterations
    if show_acceptance_ratio:
        print(f'Acceptance ratio is {acceptance_ratio}')
    return position, acceptance_ratio

#%%
def set_step_size(error_tol, dimension,
                 condition_number, upper_hessian_bound, lower_hessian_bound):
    
#    return np.power(dimension, -0.125) * 1/3
#    return np.power(dimension, -0.25) * 1/3 #big
     return np.power(dimension, -0.5) * 1/3
     

def set_leapfrog_steps(error_tol, dimension,
                       condition_number, upper_hessian_bound, lower_hessian_bound):
#    return np.int(np.ceil((dimension**0.125) * condition_number**0.25)) #aggresive
    return np.int(np.ceil(dimension**0.75/condition_number**0.75)) 
#%%

def states(initial_distribution, step_size, max_iterations = 10000,
           grad_potential = None, potential = None, target_density = None,
           flow_time=1.0, leapfrog_steps = None,
           show_acceptance_ratio = True):
    
    if leapfrog_steps == None:
        leapfrog_steps = max(int(flow_time/step_size), 1)
    
    position = initial_distribution.rvs()
    positions = [position]
    
    dimension = position.shape[0]
    acceptances = 0
    
    for j in range(max_iterations):
        
        velocity = np.random.randn(dimension)
        
        #We store the value of grad to reduce the number of computations
        grad_position = grad_potential(position)
        
        position, grad_position, acceptances = update(position, velocity, grad_position,
                                                      grad_potential, potential,
                                                      leapfrog_steps, step_size,
                                                      acceptances)
        
        positions.append(position)
        
    if show_acceptance_ratio:
        print(f'Acceptance ratio is {acceptances/max_iterations}')
    return np.array(positions)

#%%
    

def three_fourth_quantile_mixing(step_size,initial_distribution,
                                 potential, grad_potential, target_density,
                                 target_quantile, mixing_dimension = 0,
                                 max_iterations = 10000, error_tol = 0.2,
                                 flow_time = 1.0, leapfrog_steps = None):

    if leapfrog_steps == None:
        leapfrog_steps = int(flow_time/step_size)

    position = initial_distribution.rvs()
    dimension = position.shape[0]
    acceptances = 0
    
    sorted_states = [position[mixing_dimension]]
    
    for i in range(max_iterations):
        
        velocity = np.random.randn(dimension)
        
        #We store the value of grad to reduce the number of computations
        grad_position = grad_potential(position)
        
        position, grad_position, acceptances = update(position, velocity, grad_position,
                                                      grad_potential, potential,
                                                      leapfrog_steps, step_size,
                                                      acceptances)
        
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