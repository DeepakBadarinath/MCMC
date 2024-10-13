# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:31:06 2021

@author: deepa
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import HMC
#%%
normal_density = lambda x : np.exp( -(x.T @ x) / 2)
#%%
def sample_from_difference_of_Gaussians(mean_1, mean_2):
    #Sample from (N(mean_1,Id) - N(mean_2,Id))_+
    reject = True
    no_of_rejections = -1
    while reject:
        
        sample_1 = np.random.normal(loc = mean_1)
        if np.sum((sample_1-mean_1)**2) < np.sum((sample_1-mean_2)**2):
            reject = False
            
        no_of_rejections+=1
            
    return sample_1
#%%
def no_of_freedman_diaconis_bins(xs):
    xs = np.array(xs)
    q25, q75 = np.percentile(xs, [0.25, 0.75])
    bin_width = 2 * (q75 - q25) * len(xs) ** (-1/3)
    bins = int(round((xs.max() - xs.min()) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    return int(bins)

def plot_histogram(xs):
    
    sns.distplot(xs, bins=no_of_freedman_diaconis_bins(xs), kde=True)
    plt.show()

#%%
sampled_values = []
mean_1 = 1
mean_2 = -1
for i in range(10000000):
    sampled_values.append(sample_from_difference_of_Gaussians(mean_1, mean_2))

bins = no_of_freedman_diaconis_bins(sampled_values)
plt.hist(sampled_values,bins=bins,density = True)
zs = np.linspace(-10,10,10000)
plt.plot(zs,stats.norm.pdf(zs,1.0))
plt.plot(zs,stats.norm.pdf(zs,-1.0))
plt.show()

#%%
def alpha(v, new_mean):
    
    return min(1, (normal_density(v + new_mean)/normal_density(v)))

def get_one_shot_coupling_Gaussian(new_mean):
    #returns (X,Y) such that the marginals are Gaussian and P(X \neq Y) = TV(N(0,Id),N(mean,Id)))
    dimension = new_mean.shape[0]
    X = np.random.normal(loc=np.zeros(dimension))
    U = np.random.uniform()
    if U < alpha(X, new_mean):
        Y = X + new_mean
    else:
        Y = sample_from_difference_of_Gaussians(np.zeros(dimension), new_mean)
    
    return X,Y
#%%
    
xs = []
ys = []

for i in range(10000000)  :
    (x,y) = get_one_shot_coupling_Gaussian(np.array([1.0]))
    xs.append(x)
    ys.append(y)

xs = np.array(xs)
ys = np.array(ys)
#%%
plot_histogram(xs)
plot_histogram(ys)
plt.scatter(xs,ys)
#%%
no_of_iterations = 100000
x = np.zeros(1)
y = np.array([5])
dimension = 1

for i in range(no_of_iterations):
    velocity = np.random.nor
    x = HMC.update()

