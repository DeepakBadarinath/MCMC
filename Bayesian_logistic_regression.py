# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 07:39:50 2021

@author: deepa
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import multivariate_normal
import ULA
import HMC
import MRW
import MALA

#%%


#Logistic Regression - No preconditioning 
DIMENSION = 2
NO_OF_SAMPLED_POINTS = 50

X = 2 * (np.random.rand(NO_OF_SAMPLED_POINTS,DIMENSION)>0.5) - 1
X = X / np.sqrt(DIMENSION)

THETA_TRUE = np.ones((DIMENSION,1))
Y = np.exp(X @ THETA_TRUE) / (1 + np.exp(X @ THETA_TRUE))

SIGMA_X = 1/NO_OF_SAMPLED_POINTS * X.T @ X
ALPHA = 0.1

#%%

POTENTIAL = lambda theta : np.squeeze(-Y.T @ X @ theta 
                              + ALPHA * theta.T @ SIGMA_X @ theta 
                              + np.sum(np.log(1 + np.exp(X @ theta))))
#gradf = lambda theta : -X.T @ Y + ALPHA * SIGMA_X @ theta + (np.sum(X/(1 + np.exp(-X @ theta)),axis=0).reshape((-1,1)))

GRAD_POTENTIAL_CONSTANT = -X.T @ Y 

def GRAD_POTENTIAL(theta):
    theta_column = theta.reshape((-1,1))
    return np.squeeze(GRAD_POTENTIAL_CONSTANT + ALPHA * SIGMA_X @ theta_column 
                      + (np.sum(X/(1 + np.exp(-X @ theta_column)),axis=0).reshape((-1,1))))

TARGET_DENSITY = lambda theta : np.exp(-POTENTIAL(theta))

#%%
eigvals = np.linalg.eigvalsh(SIGMA_X)
UPPER_HESSIAN_BOUND = (0.25 * NO_OF_SAMPLED_POINTS + ALPHA) * eigvals[-1]
LOWER_HESSIAN_BOUND = ALPHA * eigvals[0]
#%%
def draw_paths_until_fixed_iteration(algo_states,
                                potential, grad_potential, target_density,
                                initial_distribution, step_size,
                                max_iterations, no_of_sampled_paths=1,
                                save=False, name = ''):
    
    sampled_paths = []
    
    #sampled_paths.shape =  (no_of_paths,no_of_iterations,dimension)

    for path_no in range(no_of_sampled_paths):
        
        path = algo_states(initial_distribution, 
                           step_size, max_iterations,
                           grad_potential, potential,
                           target_density)
        sampled_paths.append(path)
    
    sampled_paths = np.array(sampled_paths)
    
    if save:
        np.save(name, sampled_paths)
    
    
    
    return sampled_paths

#%%
INITIAL_DISTRIBUTION = multivariate_normal(cov= 1/UPPER_HESSIAN_BOUND * np.eye(DIMENSION))
max_iterations = 4000

step_size = 0.01
no_of_sampled_paths = 50
sampled_paths = draw_paths_until_fixed_iteration(ULA.states,
                                                 POTENTIAL, GRAD_POTENTIAL, 
                                                 TARGET_DENSITY,
                                                 INITIAL_DISTRIBUTION, step_size,
                                                 max_iterations,no_of_sampled_paths,
                                                 name = 'ULA')

#%%


