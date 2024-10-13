# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 19:29:25 2021

@author: deepa
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import multivariate_normal
import bisect
import ULA
import MALA
import MRW
import HMC
import mHMC

import scipy.stats

#%%
#Multivariate gaussian case
DIMENSIONS = np.arange(5,25,2)

diagonal_entries_uniform = lambda d : np.linspace(1,2,d)
diagonal_entries_small = lambda d : np.array([2.0] + list(np.ones(d-1)))

potential = lambda x, sigma_inv : 0.5 * x.T @ sigma_inv @ x
grad_potential = lambda x, sigma_inv : sigma_inv @ x


#correct upto proportionality
target_density = lambda x, sigma_inv : np.exp( -(x.T @ sigma_inv @ x) / 2)
#%%
#Chen potential
potential_chen = lambda x, L, m, theta : L/2 * (x[:-1].T @ x[:-1]) + (
                                            m/2 * (x[-1]**2)) - (
                                            1 / (2 * np.power(x.shape[0]-1, 0.5 - 2*theta))) * (
                                                    np.sum(np.cos(np.power(x.shape[0]-1, 0.25 - theta) * (L**(1/2))*x[:-1])))

grad_potential_chen = lambda x, L, m, theta : np.append(L * x[:-1] + (L**0.5)/2 * np.power(x.shape[0]-1,theta - 0.25) * (
                                            np.sin(np.power(x.shape[0]-1, 0.25-theta)*(L**0.5)*x[:-1])), m*x[-1])

target_density_chen = lambda x, L, m, theta : np.exp(-potential_chen(x,L,m,theta))
#%%
UPPER_HESSIAN_BOUND  = 1
#LOWER_HESSIAN_BOUND = 0.25
# LOWER_HESSIAN_BOUND = 0.5
LOWER_HESSIAN_BOUND = 1
CONDITION_NUMBER = UPPER_HESSIAN_BOUND / LOWER_HESSIAN_BOUND

#%%

TARGET_QUANTILE  = scipy.stats.norm.ppf(0.75,scale=2)
error_tol = 0.2
EASY_TARGET_QUANTILE = scipy.stats.norm.ppf(0.75,scale=np.sqrt(2))
TARGET_QUANTILE_CHEN = scipy.stats.norm.ppf(0.75,scale=1.0)

#%%
EMPIRICAL_SIZE = 250000
#EMPIRICAL_SAMPLES = 2 * multivariate_normal.rvs(size = EMPIRICAL_SIZE)
EMPIRICAL_SAMPLES = multivariate_normal.rvs(size = EMPIRICAL_SIZE)
#%%
def cumulative_three_fourth_quantile(states):
    """
    Given a 1D array finds cumulative/running three fourth quantile.
    """
    sorted_states = []
    
    for i in range(len(states)):
                
        bisect.insort(sorted_states,states[i])
        
        #len(sorted_states) = i+1
        if (i+1) % 4 == 0:
            empirical_quantile = sorted_states[int(0.75 * (i+1)) - 1]
        else:
            
            fraction = 0.75 * (i+1)
            
            #We subract 1 due to 0 indexing
            
            n = int(fraction) - 1
            
            #fractions lies bw n+1 and n+2
            
            empirical_quantile = (sorted_states[n]*(n+2-fraction)
                                  + sorted_states[n+1]*(fraction-n-1))

    
    return i


#%%
def three_fourth_quantile_mixing_time_for_states(states,
                                                 target_quantile, 
                                                 error_tol = 0.2):
    
    """ 
    Given a 1D vector of states determines mixing time for
    three fourth quantile
    """
    
    sorted_states = []
    
    for i in range(len(states)):
                
        bisect.insort(sorted_states,states[i])
        
        #len(sorted_states) = i+1
        if (i+1) % 4 == 0:
            empirical_quantile = sorted_states[int(0.75 * (i+1)) - 1]
        else:
            
            fraction = 0.75 * (i+1)
            
            #We subract 1 due to 0 indexing
            
            n = int(fraction) - 1
            
            #fractions lies bw n+1 and n+2
            
            empirical_quantile = (sorted_states[n]*(n+2-fraction)
                                  + sorted_states[n+1]*(fraction-n-1))
            
        if np.abs(empirical_quantile - target_quantile) < error_tol:
            return i
    
    return i
#%%

def min_max_mixing_time_for_distribution(batch_size = 10000,
                                         empirical_size = 10000,
                                         error_tol = 0.2):
    
    mixing_times = []
    
    for i in range(batch_size):
    
        empirical_samples = 2 * multivariate_normal.rvs(size = empirical_size)
    
        mixing_time = three_fourth_quantile_mixing_time_for_states(empirical_samples,
                                                                   TARGET_QUANTILE,
                                                                   error_tol)
        mixing_times.append(mixing_time)
    
    mixing_times = np.array(mixing_times)
    
    return min(mixing_times), max(mixing_times), sum(mixing_times) / len(mixing_times), np.std(mixing_times)

#%%

_,_,AVG_MIXING_TIME,STD_MIXING_TIME = min_max_mixing_time_for_distribution(error_tol=0.4)
print(AVG_MIXING_TIME, STD_MIXING_TIME)

#%%
def draw_paths_until_fixed_iteration(algo_states,
                                    potential, grad_potential, target_density,
                                    initial_distribution, step_size, flow_time = 1.0,
                                    max_iterations=10000, no_of_sampled_paths=1,
                                    save=False, name = ''):
    
    sampled_paths = []
    
    #sampled_paths.shape =  (no_of_paths,no_of_iterations,dimension)

    for path_no in range(no_of_sampled_paths):
        
        path = algo_states(initial_distribution, 
                           step_size, max_iterations,
                           grad_potential, potential,
                           target_density, flow_time)
        sampled_paths.append(path)
    
    sampled_paths = np.array(sampled_paths)
    
    if save:
        np.save(name, sampled_paths)
    
    
    
    return sampled_paths


#%%
no_of_sampled_paths = 10
algo = MALA
flow_time = 0.5
#Checing histogram and convg etc

for dimension in [5]:
    
    diagonal_entries = diagonal_entries_small(dimension)
    sigma_inv = np.diag(1 / diagonal_entries)
    
    potential_given_cov = lambda x : potential(x,sigma_inv)
    grad_potential_given_cov = lambda x: grad_potential(x,sigma_inv)
    target_density_given_cov = lambda x: target_density(x,sigma_inv)
    
    initial_distribution = multivariate_normal(cov= 1 * np.eye(dimension))
    
    step_size = algo.set_step_size(error_tol, dimension,
                                  CONDITION_NUMBER, UPPER_HESSIAN_BOUND, 
                                  LOWER_HESSIAN_BOUND)
    
    sampled_paths = draw_paths_until_fixed_iteration(algo.states,
                                                     potential_given_cov, 
                                                     grad_potential_given_cov,
                                                     target_density_given_cov,
                                                     initial_distribution, step_size,
                                                     max_iterations = 500,
                                                     no_of_sampled_paths = no_of_sampled_paths,
                                                     save = False,
                                                     name = algo.NAME + f'_{dimension}')

#%%
#Plotting histograms

NO_OF_BINS = 100

#Visual check
all_points = sampled_paths[:,:,0].flatten()

_,_,_ = plt.hist(all_points, bins=NO_OF_BINS, range=[-7.5,7.5], 
                         density=True, label = 'eHMC', alpha = 0.7 )
_,_,_ = plt.hist(EMPIRICAL_SAMPLES, bins=NO_OF_BINS, range=[-7.5,7.5], 
                     density=True, label = 'OG', alpha = 0.7)

plt.title("Histogram")
plt.legend()
plt.show()


def find_density_of_interval(sorted_points,interval_start,interval_end):
    no_of_points_less_than_start = bisect.bisect_left(sorted_points, interval_start)
    no_of_points_less_than_end = bisect.bisect_left(sorted_points, interval_end)
    return ((no_of_points_less_than_end - no_of_points_less_than_start) 
            / len(sorted_points))
    

def errors_sum(algo_points, empirical_points, error_bins):
    
    sorted_algo_points = sorted(algo_points)
    sorted_empirical_points = sorted(empirical_points)
    
    error_sum = 0
    
    for i in range(1,len(error_bins)):
        
        error_sum += abs(find_density_of_interval(sorted_algo_points, error_bins[i-1], error_bins[i]) 
                    - find_density_of_interval(sorted_empirical_points, error_bins[i-1], error_bins[i]))
        
        return error_sum


errors_sum(all_points,EMPIRICAL_SAMPLES,[-4,-2,0,2,4])
#%%
#Acceptance ratio VS Dimension
import mHMC
flow_time = 1.0
sample_no = 500

accept_ratio_avgs = []
accept_ratio_stds = []

algo = mHMC
DIMENSIONS = np.arange(5,25,2)

for dimension in DIMENSIONS:
    diagonal_entries = np.array([2.0] + list(np.ones(dimension-1)))
    sigma_inv = np.diag(1 / diagonal_entries)
    
    potential_given_cov = lambda x : potential(x,sigma_inv)
    grad_potential_given_cov = lambda x: grad_potential(x,sigma_inv)
    target_density_given_cov = lambda x: target_density(x,sigma_inv)
    
    initial_distribution = multivariate_normal(cov=np.eye(dimension),
                                               allow_singular = True)
    
    step_size = algo.set_step_size(error_tol, dimension,
                                  CONDITION_NUMBER, UPPER_HESSIAN_BOUND,
                                  LOWER_HESSIAN_BOUND)
    
    accept_ratio_avg = 0
    accept_ratio_std = 0
    
    for i in range(sample_no):
        _, acceptance_ratio = algo.final_state(initial_distribution, step_size,
                                               300,
                                               grad_potential_given_cov,
                                               potential_given_cov,
                                               target_density_given_cov)
        
        new_accept_ratio_avg = accept_ratio_avg + (acceptance_ratio - accept_ratio_avg)/(i+1)
        
        accept_ratio_std = (i*(accept_ratio_std  + (new_accept_ratio_avg - accept_ratio_avg)**2) + 
                             (acceptance_ratio - new_accept_ratio_avg)**2)
        
        accept_ratio_std = np.sqrt(accept_ratio_std/(i+1))
        
        accept_ratio_avg = new_accept_ratio_avg
    
    print(f"dim = {dimension}, avg accept ratio is {accept_ratio_avg}, emp std is {accept_ratio_std}")

    accept_ratio_avgs.append(accept_ratio_avg)
    accept_ratio_stds.append(accept_ratio_std)

saved_dim_string = algo.NAME + '_avg_accept_ratio_dim_one_fourth.npy'
np.save(saved_dim_string, accept_ratio_avgs)

saved_std_string = algo.NAME + '_std_accept_ratio_dim_one_fourth.npy'
np.save(saved_std_string, accept_ratio_stds)

#%%

mHMC_ar_dim_onefourth_avg = np.load('mHMC_avg_accept_ratio_dim_one_fourth.npy')
mHMC_ar_dim_onefourth_std = np.load('mHMC_std_accept_ratio_dim_one_fourth.npy')

#%%
stepsizes_list = ['$h = d^{-1/2}$', '$h = d^{-1}$', '$h = d^{-1/4}$']

fig,ax = plt.subplots(figsize=(9,7))
mala_ars_avg = np.load('acceptance_ratio_avg_dim_mala_one_half_one_one_fourth.npy')
mala_ars_std = np.load('acceptance_ratio_std_dim_mala_one_half_one_one_fourth.npy')

DIMENSIONS = np.arange(5,50,2)
point_colors = ['blue','darkgreen','crimson']
line_colors = ['blue','darkgreen','crimson']

slopes = []

markers = ["*", "+", "x", "s"]
marker_sizes = (mala_ars_std)**2 * 40
#marker_size_hmc = (std_for_all_algos[-1])**2 * np.pi / 20
#marker_sizes[-1] = marker_size_hmc

title_string = f'Acceptance ratio vs dimension - MALA'
ax.set_title(title_string)

for i,algo_name in enumerate(stepsizes_list):
    
    ax.scatter(DIMENSIONS, mala_ars_avg[i],
               c = point_colors[i], label=algo_name, 
               s = marker_sizes[i],
               marker = markers[i],
               alpha = 0.7)
    
    ax.plot(DIMENSIONS, mala_ars_avg[i],
            c = point_colors[i])
    

ax.set_xlabel('Dimension')
ax.set_ylabel('Acceptance Ratio')

f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))


ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(g))
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(g))


lgnd = ax.legend(loc="bottom left", scatterpoints=1, fontsize=10)

for i,_ in enumerate(stepsizes_list):
    lgnd.legendHandles[i]._sizes = [30]
    
plt.savefig('acceptance_ratio_vs_dimension_mala.png')

plt.show()


#%%
# Mixing time VS Dimension
import MALA

flow_time = 1.0
sample_no = 10000
error_tol = 0.4

algo = MALA

mixing_times_for_algo = []
std_for_algo = []

DIMENSIONS = np.arange(5,11)
#for dimension in DIMENSIONS:
for dimension in DIMENSIONS:
    
    #diagonal_entries = diagonal_entries_small(dimension)
    '''
    diagonal_entries = np.array([2.0] + list(np.ones(dimension-1)))
    sigma_inv = np.diag(1 / diagonal_entries)
    
    potential_given_cov = lambda x : potential(x,sigma_inv)
    grad_potential_given_cov = lambda x: grad_potential(x,sigma_inv)
    target_density_given_cov = lambda x: target_density(x,sigma_inv)'''
    
    L = 1
    m = 1
    theta = 1/40
    potential_given_params = lambda x : potential_chen(x,L,m,theta)
    grad_potential_given_params = lambda x: grad_potential_chen(x,L,m,theta)
    target_density_given_params = lambda x: target_density_chen(x,L,m,theta)
    
    initial_distribution = multivariate_normal(cov=np.eye(dimension)*0,
                                               allow_singular = True)
    
    step_size = algo.set_step_size(error_tol, dimension,
                                  CONDITION_NUMBER, UPPER_HESSIAN_BOUND, 
                                  LOWER_HESSIAN_BOUND)
    
#    leapfrog_steps = algo.set_leapfrog_steps(error_tol, dimension, CONDITION_NUMBER, UPPER_HESSIAN_BOUND, LOWER_HESSIAN_BOUND)
    leapfrog_steps = None
    
    mixing_time_avg = 0
    mixing_time_std = 0
    
    for i in range(sample_no):
        
        mixing_time_sample = algo.three_fourth_quantile_mixing(step_size,
                                                              initial_distribution,
                                                              potential_given_params,
                                                              grad_potential_given_params,
                                                              target_density_given_params,
                                                              TARGET_QUANTILE_CHEN,
                                                              error_tol=error_tol,
                                                              max_iterations = 10000,
                                                              mixing_dimension = -1)
        
        new_mixing_time_avg = mixing_time_avg + (mixing_time_sample - mixing_time_avg)/(i+1)
        
        mixing_time_std = (i*(mixing_time_std  + (new_mixing_time_avg - mixing_time_avg)**2) + 
                             (mixing_time_sample - new_mixing_time_avg)**2)
        
        mixing_time_std = np.sqrt(mixing_time_std/(i+1))
        
        mixing_time_avg = new_mixing_time_avg
        
    if algo == HMC:
        mixing_time_avg = mixing_time_avg * flow_time/step_size
        mixing_time_std = mixing_time_std * (flow_time/step_size)**2
        
    elif algo == mHMC and leapfrog_steps != None:
        mixing_time_avg = mixing_time_avg * leapfrog_steps * 2
        mixing_time_std = mixing_time_std * (leapfrog_steps**2) * 4
    
    elif algo == mHMC and leapfrog_steps == None:
        mixing_time_avg = mixing_time_avg * flow_time/step_size * 2
        mixing_time_std = mixing_time_std * (flow_time/step_size)**2 * 4
        
    print(f"dim = {dimension}, avg mix time is {mixing_time_avg}, emp std is {mixing_time_std}")
    
    mixing_times_for_algo.append(mixing_time_avg)
    std_for_algo.append(mixing_time_std)

saved_dim_string = algo.NAME + '_slow_chen_mixing_time_vs_dimension_nonwarm.npy'
np.save(saved_dim_string, mixing_times_for_algo)

saved_std_string = algo.NAME + '_slow_chen_std_dim_time_nonwarm.npy'
np.save(saved_std_string, std_for_algo)

m,b = np.polyfit(np.log(DIMENSIONS),np.log(np.array(mixing_times_for_algo)),1)
print(m,b)
#%%
#Non warm loading values
ula_time = np.load('ula_chen_mixing_time_vs_dimension_nonwarm.npy')
ula_std = np.load('ula_chen_std_dim_time_nonwarm.npy')
mala_time = np.load('mala_chen_mixing_time_vs_dimension_nonwarm.npy')
mala_std = np.load('mala_chen_std_dim_time_nonwarm.npy')
uhmc_time = np.load('hmc_chen_mixing_time_vs_dimension_nonwarm.npy')
uhmc_std = np.load('hmc_chen_std_dim_time_nonwarm.npy')
mrw_time = np.load('mrw_chen_mixing_time_vs_dimension_nonwarm.npy')
mrw_std = np.load('mrw_chen_std_dim_time_nonwarm.npy')
mhmc_time = np.load('mhmc_chen_mixing_time_vs_dimension_nonwarm.npy')
mhmc_std = np.load('mhmc_chen_std_dim_time_nonwarm.npy')
mhmc_big_time = np.load('mhmc_big_chen_mixing_time_vs_dimension_nonwarm.npy')
mhmc_big_std = np.load('mhmc_big_chen_std_dim_time_nonwarm.npy')
nuts_time = np.load('nuts_chen_mixing_time_vs_dimension_nonwarm.npy')/8
nuts_std = np.load('nuts_chen_std_dim_time_nonwarm.npy')

mixing_times_for_all_algos = np.array([ula_time] + [mrw_time] + 
                                      [mala_time] + [uhmc_time] + [mhmc_time] + 
                                      [nuts_time])
np.save('mixing_time_vs_dim_ula_mrw_mala_uhmc_mhmc_nuts_non_warm_chen.npy', mixing_times_for_all_algos)

std_for_all_algos = np.array([ula_std] + [mrw_std] + 
                             [mala_std] + [uhmc_std] + [mhmc_std] + 
                             [nuts_std])
np.save('std_vs_dim_ula_mrw_mala_uhmc_mhmc_nuts_non_warm_chen.npy', std_for_all_algos)

#%%
#Warm loading values
mixing_times_for_all_algos = np.vstack([np.load('mixing_time_vs_dim_ula_mrw_mala_uhmc_mhmc_big_fast.npy'),
                                        np.load('nuts_mixing_time_vs_dimension.npy')])
std_for_all_algos = np.vstack([np.load('std_vs_dim_ula_mrw_mala_uhmc_mhmc_big_fast.npy'),
                               np.load('nuts_std_dim_time.npy')])

np.save('mixing_time_vs_dim_ula_mrw_mala_uhmc_mhmc_mhmc_big_nuts_fast.npy', mixing_times_for_all_algos)
np.save('std_vs_dim_ula_mrw_mala_uhmc_mhmc_mhmc_big_nuts_fast.npy', std_for_all_algos)


#%%
algo_names = ['ULA', 'MRW', 'MALA', 'uHMC', 'mHMC', 'NUTS']
#algo_names = ['MALA', 'uHMC', 'mHMC', 'mHMC_big']

fig,ax = plt.subplots(figsize=(9,7))
mixing_times_for_all_algos = np.load('mixing_time_vs_dim_ula_mrw_mala_uhmc_mhmc_nuts_non_warm_chen.npy')
#mixing_times_for_all_algos = mixing_times_for_all_algos[2:]

std_for_all_algos = np.load('std_vs_dim_ula_mrw_mala_uhmc_mhmc_nuts_non_warm_chen.npy')
#std_for_all_algos = std_for_all_algos[2:]

DIMENSIONS = np.arange(5,11)
point_colors = ['blue','darkgreen','crimson','purple', 'saddlebrown','hotpink']
line_colors = ['blue','darkgreen','crimson','purple', 'saddlebrown','hotpink']

slopes = []

marker_sizes = (std_for_all_algos)**2 * np.pi 
marker_size_mhmc = (std_for_all_algos[-2])**2/50
#marker_size_mhmc_big = (std_for_all_algos[-1])**2 * np.pi * 2
marker_size_uhmc = (std_for_all_algos[-3])**2 * np.pi * 1/10
marker_sizes[-2] = marker_size_mhmc/15
marker_sizes[-3] = marker_size_uhmc/200


title_string = 'Computational cost VS Dimension'
ax.set_title(title_string)

for i,algo_name in enumerate(algo_names):
    
    m,b = np.polyfit(np.log(DIMENSIONS),np.log(mixing_times_for_all_algos[i]),1)
    ax.scatter(DIMENSIONS,mixing_times_for_all_algos[i],
               c = point_colors[i],label=algo_name + f' slope = {np.round(m,2)}', 
               s = marker_sizes[i],
               alpha = 0.7)
    m,b = np.polyfit(np.log(DIMENSIONS),np.log(mixing_times_for_all_algos[i]),1)
    slopes.append(m)
    ax.plot(DIMENSIONS,np.power(DIMENSIONS,m)*np.exp(b),color = line_colors[i],
            alpha = 0.6, linewidth = 1.5)
    

ax.set_xlabel('Dimension')
ax.set_ylabel('# of function and gradient evaluations')
ax.set_xscale('log')
ax.set_yscale('log')


f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))


ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(g))
ax.set_xticks(np.array([8,10]))

ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(g))
ax.set_yticks(np.arange(80,1400,200))
ax.grid(True)


lgnd = ax.legend(loc="upper left", scatterpoints=1, fontsize=10)

for i,_ in enumerate(algo_names):
    lgnd.legendHandles[i]._sizes = [30]

plt.savefig('tmix_vs_dimension_all_algos_chen.png')

plt.show()
#%%
import MALA
sample_no = 10000
ERROR_TOLS = np.array([0.3,0.35,0.4,0.45,0.5])
dimension = 5
flow_time = 1.0

'''diagonal_entries = np.array([2.0] + list(np.ones(dimension-1)))
#diagonal_entries = np.array([4.0] + list(np.ones(dimension-1)))
sigma_inv = np.diag(1 / diagonal_entries)


potential_given_cov = lambda x : potential(x,sigma_inv)
grad_potential_given_cov = lambda x: grad_potential(x,sigma_inv)
target_density_given_cov = lambda x: target_density(x,sigma_inv)'''

L = 1
m = 1
theta = 1/40
potential_given_params = lambda x : potential_chen(x,L,m,theta)
grad_potential_given_params = lambda x: grad_potential_chen(x,L,m,theta)
target_density_given_params = lambda x: target_density_chen(x,L,m,theta)

initial_distribution = multivariate_normal(cov=np.eye(dimension)*0,
                                           allow_singular = True)

step_size = algo.set_step_size(error_tol, dimension,
                              CONDITION_NUMBER, UPPER_HESSIAN_BOUND, 
                              LOWER_HESSIAN_BOUND)
algo = mHMC

mixing_times_for_algo = []
std_for_algo = []

for error_tol in ERROR_TOLS:
    
    #diagonal_entries = diagonal_entries_small(dimension)
    
    step_size = algo.set_step_size(error_tol, dimension,
                                  CONDITION_NUMBER, UPPER_HESSIAN_BOUND, 
                                  LOWER_HESSIAN_BOUND)
    
    mixing_time_avg = 0
    mixing_time_std = 0
    
    for i in range(sample_no):
        
        mixing_time_sample = algo.three_fourth_quantile_mixing(step_size,
                                                              initial_distribution,
                                                              potential_given_params,
                                                              grad_potential_given_params,
                                                              target_density_given_params,
                                                              TARGET_QUANTILE_CHEN,
                                                              error_tol=error_tol,     
                                                              max_iterations = 10000)
        
        new_mixing_time_avg = mixing_time_avg + (mixing_time_sample - mixing_time_avg)/(i+1)
        
        mixing_time_std = (i*(mixing_time_std  + (new_mixing_time_avg - mixing_time_avg)**2) + 
                             (mixing_time_sample - new_mixing_time_avg)**2)
        
        mixing_time_std = np.sqrt(mixing_time_std/(i+1))
        
        mixing_time_avg = new_mixing_time_avg
       
    if algo == HMC:
        mixing_time_avg = mixing_time_avg * flow_time/step_size
        mixing_time_std = mixing_time_std * (flow_time/step_size)**2
        
    elif algo == mHMC and leapfrog_steps != None:
        mixing_time_avg = mixing_time_avg * leapfrog_steps * 2
        mixing_time_std = mixing_time_std * (leapfrog_steps**2) * 4
    
    elif algo == mHMC and leapfrog_steps == None:
        mixing_time_avg = mixing_time_avg * flow_time/step_size * 2
        mixing_time_std = mixing_time_std * (flow_time/step_size)**2 * 4
        
        
    print(f"error tol = {error_tol}, avg mix time is {mixing_time_avg}, emp std is {mixing_time_std}")
    mixing_times_for_algo.append(mixing_time_avg)
    std_for_algo.append(mixing_time_std)


saved_dim_string = algo.NAME + '_mixing_time_tol_chen.npy'
np.save(saved_dim_string, mixing_times_for_algo)

saved_std_string = algo.NAME + '_std_tol_chen.npy'
np.save(saved_std_string, std_for_algo)

#%%

ula_time = np.load('ula_mixing_time_tol_chen.npy')
ula_std = np.load('ula_std_tol_chen.npy')
mala_time = np.load('mala_mixing_time_tol_chen.npy')
mala_std = np.load('mala_std_tol_chen.npy')
uhmc_time = np.load('hmc_mixing_time_tol_chen.npy')
uhmc_std = np.load('hmc_std_tol_chen.npy')
mrw_time = np.load('mrw_mixing_time_tol_chen.npy')
mrw_std = np.load('mrw_std_tol_chen.npy')
mhmc_time = np.load('mhmc_mixing_time_tol_chen.npy')
mhmc_std = np.load('mhmc_std_tol_chen.npy')
nuts_time = np.load('nuts_chen_mixing_time_vs_tol_nonwarm.npy')/8
nuts_std = np.load('nuts_chen_std_tol_nonwarm.npy')
mixing_times_for_all_algos = np.array([ula_time] + [mrw_time] + 
                                      [mala_time] + [uhmc_time] + [mhmc_time] + 
                                      [nuts_time])
np.save('mixing_time_vs_tol_ula_mrw_mala_uhmc_mhmc_nuts_non_warm_chen.npy', mixing_times_for_all_algos)

std_for_all_algos = np.array([ula_std] + [mrw_std] + 
                             [mala_std] + [uhmc_std] + [mhmc_std] + 
                             [nuts_std])
np.save('std_vs_tol_ula_mrw_mala_uhmc_mhmc_nuts_non_warm_chen.npy', std_for_all_algos)
#%%
algo_names = ['ULA', 'MRW', 'MALA', 'uHMC', 'mHMC', 'NUTS']

fig,ax = plt.subplots(figsize=(9,7))
mixing_times_for_all_algos = np.load('mixing_time_vs_tol_ula_mrw_mala_uhmc_mhmc_nuts_non_warm_chen.npy')
std_for_all_algos = np.load('std_vs_tol_ula_mrw_mala_uhmc_mhmc_nuts_non_warm_chen.npy')

ERROR_TOLS = np.array([0.3,0.35,0.4,0.45,0.5])
INV_ERROR_TOLS = 1 / ERROR_TOLS
point_colors = ['blue','darkgreen','crimson','purple', 'darkgray', 'hotpink']
line_colors = ['blue','darkgreen','crimson','purple', 'darkgray', 'hotpink']

slopes = []

marker_sizes = (std_for_all_algos)**2 * np.pi
marker_size_mhmc = (std_for_all_algos[-2])**2 * np.pi / 50
marker_sizes[-2] = marker_size_mhmc
marker_size_uhmc = (std_for_all_algos[-3])**2 * np.pi / 2
marker_sizes[-3] = marker_size_uhmc

title_string = 'Computational cost VS Inverse error tolerance'
ax.set_title(title_string)

for i,algo_name in enumerate(algo_names):
    
    m,b = np.polyfit(np.log(INV_ERROR_TOLS), np.log(mixing_times_for_all_algos[i]),1)
    ax.scatter(INV_ERROR_TOLS, mixing_times_for_all_algos[i],
               c = point_colors[i], label=algo_name + f' slope = {np.round(m,2)}', 
               s = marker_sizes[i],
               alpha = 0.7)
    m,b = np.polyfit(np.log(INV_ERROR_TOLS), np.log(mixing_times_for_all_algos[i]),1)
    slopes.append(m)
    ax.plot(INV_ERROR_TOLS, np.power(INV_ERROR_TOLS,m)*np.exp(b), color = line_colors[i],
            alpha = 0.6, linewidth = 1.5)
    

ax.set_xlabel('Inverse error tolerance')
ax.set_ylabel('# of function and gradient evaluations')
ax.set_xscale('log')
ax.set_yscale('log')


f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))


ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(g))
ax.set_xticks(np.arange(2, 3.33, 0.4))

ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(g))
ax.set_yticks(np.arange(250, 1250, 200))
ax.grid(True)


lgnd = ax.legend(loc="upper left", scatterpoints=1, fontsize=10)

for i,_ in enumerate(algo_names):
    lgnd.legendHandles[i]._sizes = [30]


plt.savefig('tmix_vs_tol_chen_ula_mrw_mala_uhmc_mhmc_nuts.png')

plt.show()
#%%