# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 09:44:58 2022

@author: deepa
"""

import numpy as np
import bisect


def find_three_fourth_mixing_time_of_array(states, delta, target):
    
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
            
        if np.abs(empirical_quantile - target) < delta:
            return i
    
    return i


states = [0,1,2,3,4,5]
target = 3
delta = 1
print(find_three_fourth_mixing_time_of_array(states, delta, target))

