# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:57:20 2021

@author: deepa
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import HMC
#%%


# target density = exp(-f(X)), gradf is the gradient of this f
POTENTIAL = lambda x: 0.5 * x * x

GRAD_POTENTIAL = lambda x : x

TARGET_DENSITY()

#Constants bounding the rate of growth of the Hessian of Potential
UPPER_HESSIAN_BOUND  = 1
LOWER_HESSIAN_BOUND = 0.5
CONDITION_NUMBER = UPPER_HESSIAN_BOUND / LOWER_HESSIAN_BOUND

DIMENSION = 2

#%%

l = [1,3,4]
l.append([])
l
l.append(45)
l