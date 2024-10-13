# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 09:51:20 2021

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
#import mHMC_Gaussian as mHMC
#import exactHMC_Gaussian as eHMC
import scipy.stats
#%%
mixing_times_for_initial_algos = np.load('mixing_time_vs_dim_ula_mrw_mala.npy')
mixing_times_for_uhmc = np.load('mixing_time_vs_dim_hmc.npy')
mixing_times_for_all_algos = np.load('mixing_time_vs_dim_ula_mrw_mala_uhmc.npy')

#%%


fig,ax = plt.subplots(figsize=(9,7))
algo_names = ['ULA','MRW','MALA','uHMC']

point_colors = ['blue','darkgreen','crimson','purple']
line_colors = ['blue','darkgreen','crimson','purple']
slopes = []

marker_sizes = (std_for_all_algos)**2 * np.pi * 5
marker_size_hmc = (std_for_all_algos[-1])**2 * np.pi / 20
marker_sizes[-1] = marker_size_hmc

for i,algo_name in enumerate(algo_names):
    
    inv_error_tols = 1 / ERROR_TOLS
    m,b = np.polyfit(np.log(inv_error_tols),np.log(mixing_times_for_all_algos[i]),1)
    ax.scatter(inv_error_tols, mixing_times_for_all_algos[i],
               c = point_colors[i], label=algo_name + f' slope = {np.round(m,2)}', 
               s = marker_sizes[i],
               alpha = 0.7)
    slopes.append(m)
    ax.plot(inv_error_tols,np.power(inv_error_tols,m)*np.exp(b),color = line_colors[i],
            alpha = 0.6, linewidth = 1.5)
    
title_string = 'Mixing Time VS Inverse Error Tolerance'
ax.set_title(title_string)

ax.set_xlabel('Inverse Error Tolerance')
ax.set_ylabel('Mixing time')
ax.set_xscale('log')
ax.set_yscale('log')

#ax.set_xticks(np.arange(5,25,3))

f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1e' % x))
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(g))
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(g))
#ax.set_yticks(np.arange(50,1500,200))
ax.grid(True)



ax.legend()

plt.savefig('tmix_vs_error_tol_ula_mrw_mala_uhmc2')

