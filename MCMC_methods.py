# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 07:14:53 2021

@author: deepa
"""

import numpy as np
from MCMC_kernels import *

def metropolisHastings(target_density, proposal_kernel, initial_distribution, maxIterations = 10000):
    
    """ A general metropolis hastings method used to sample from a distribution 
    with a target density.
    
    We accept the step taken by a proposal kernel with a certain acceptance probability
    
    Parameters : target_density : function type
                 Density of the target distribution
                 
                 proposal_kernel : kernel type (from MCMC_kernels)
                 Probability kernel of the proposal
                 
                 initial_distribution : scipy.stats._continuous_distns type
                 Intial distribution of the first step of the chain
                 
                 maxIterations : int type
                 No of iterations to run the chain for
                 
    
    Outputs :   x : numpy.ndarray type
                Final state of the markov chain after maxIteration no of steps
                
                states : list type
                List of all states traversed by chain until maxIteration no of steps
                
                ratio : float type
                Acceptance ratio of the chain until maxIteration no of steps
    """
    
    
    x = initial_distribution.rvs()
    acceptances = 0
    states = [x]
    
    
    for i in range(maxIterations):
        y = proposal_kernel.rvs(x)
        
        accept_prob = min(1,(target_density(y)*proposal_kernel.pdf(y,x)) / (target_density(x)*proposal_kernel.pdf(x,y)) )
        
        U = np.random.rand()
        
        if U < accept_prob:
            
            acceptances+=1
            x = y
            
        states.append(x)
        
    return x,states,acceptances/maxIterations



def MRW(target_density,h,initial_distribution ,maxIterations = 10000):
    
    """ 
    Metropolis Random Walk
    
    Metrolpolis Hastings method with chain update
    
    X -> X + root(2h)Z
    
    """
    
    proposal_kernel = MRWKernel(h)
    return metropolisHastings(target_density, proposal_kernel , initial_distribution ,maxIterations)


def ULA(target_density,h,initial_distribution ,maxIterations = 10000,gradf=None):
    
    """
    Underdamped Langevin Algorithm 
    
    Target density is given by exp(-f(x))
    
    Chain Update:
    X -> X + root(2h)Z + h * gradf(X)
    
    
    
    """
    
    proposal_kernel = ULAKernel(h,gradf)
    x = initial_distribution.rvs()
    states = [x]
    
    for i in range(maxIterations):
        
        y = proposal_kernel.rvs(x)
        states.append(y)
        x = y
    
    return x,states

def MALA(target_density,h,initial_distribution ,maxIterations = 10000,gradf=None):
    
    proposal_kernel = ULAKernel(h,gradf)
    return metropolisHastings(target_density, proposal_kernel , initial_distribution ,maxIterations)

#%%
    
def metropolisHastingsMixing(target_density, proposal_kernel, initial_distribution, 
                             quantile, target_quantile, mixing_dimension = 0,
                             maxIterations = 10000, errorTol = 0.2):
    
    """ A general metropolis hastings method used to sample from a distribution 
    with a target density.
    
    The probabability kernel is given by proposal kernel and the initial distribution is given
    
    Quantile mixing time of the first dimension
    
    We run the chain for maxIterations no of iterations 
    
    Parameters : target_density : function type
                 Density of the target distribution
                 
                 proposal_kernel : kernel type (from MCMC_kernels)
                 
    """
    
    
    x = initial_distribution.rvs()
    import bisect
    
    sorted_mixing_states = [x[mixing_dimension]]
    
    
    for i in range(maxIterations):
        y = proposal_kernel.rvs(x)
        
        accept_prob = min(1,(target_density(y)*proposal_kernel.pdf(y,x)) / (target_density(x)*proposal_kernel.pdf(x,y)) )
        
        U = np.random.rand()
        
        if U < accept_prob:
            
            x = y
            

            bisect.insort(sorted_mixing_states,x[mixing_dimension])
            empirical_quantile = sorted_mixing_states[int(quantile * (len(sorted_mixing_states)-1))]
            
            if np.abs(empirical_quantile - target_quantile) < errorTol:
                
                #print(f"Mixing happens for dimension {mixing_dimension} in {i+1} steps for total dimension {x.shape[0]}")
                return i+1
        
        
 
    #print(f"No mixing in {maxIterations} steps for error tol = {errorTol} and total dimension {x.shape[0]}")
        
    return i+1
    


def MRWMixing(target_density,h,initial_distribution, 
              quantile, target_quantile, mixing_dimension = 0 ,
              maxIterations = 10000, errorTol = 0.2):
    
    proposal_kernel = MRWKernel(h)
    return metropolisHastingsMixing(target_density, proposal_kernel, initial_distribution, 
                             quantile, target_quantile, mixing_dimension,
                             maxIterations, errorTol)


def ULAMixing(target_density,h,initial_distribution,gradf,
              quantile, target_quantile, mixing_dimension = 0 ,
              maxIterations = 10000, errorTol = 0.2):
    
    proposal_kernel = ULAKernel(h,gradf)
    x = initial_distribution.rvs()
    
    import bisect
    sorted_mixing_states = [x[mixing_dimension]]
    
    for i in range(maxIterations):
        
        x = proposal_kernel.rvs(x)
        
        bisect.insort(sorted_mixing_states,x[mixing_dimension])
        empirical_quantile = sorted_mixing_states[int(quantile * (len(sorted_mixing_states)-1))]
            
        if np.abs(empirical_quantile - target_quantile) < errorTol:
            return i+1
    
    return i+1


def MALAMixing(target_density,h,initial_distribution,gradf,
              quantile, target_quantile, mixing_dimension = 0 ,
              maxIterations = 10000, errorTol = 0.2):
    
    proposal_kernel = ULAKernel(h,gradf)
    return metropolisHastingsMixing(target_density, proposal_kernel, initial_distribution, 
                             quantile, target_quantile, mixing_dimension,
                             maxIterations, errorTol)



