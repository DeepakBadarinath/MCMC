# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 03:40:17 2021

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
""" 
The target density is a symmetric mixture of a pair of 2D gaussians, with means on (0.5,0.5) and -(0.5,0.5)

Axes of maximal variation is the x=y line and x=-y line
We look at the distribution projected to these axes

"""
GAUSSIAN_CENTER = np.array([0.5,0.5])

TARGET_DENSITY = lambda X : (0.5 * 0.5 * 1/np.pi 
                             * (np.exp(-np.dot(X-GAUSSIAN_CENTER,X-GAUSSIAN_CENTER)/2)
                                 + np.exp(-np.dot(X+GAUSSIAN_CENTER,X+GAUSSIAN_CENTER)/2)))

# target density = exp(-f(X)), gradf is the gradient of this f
POTENTIAL = lambda X : (0.5 * np.dot(X-GAUSSIAN_CENTER,X-GAUSSIAN_CENTER) 
                        - np.log(1 + np.exp(-2 * np.dot(X,GAUSSIAN_CENTER))))

GRAD_POTENTIAL = lambda X : (X - GAUSSIAN_CENTER + 2 * GAUSSIAN_CENTER * (1/(1 + np.exp(2 * np.dot(X,GAUSSIAN_CENTER)))))


#Constants bounding the rate of growth of the Hessian of Potential
UPPER_HESSIAN_BOUND  = 1
LOWER_HESSIAN_BOUND = 0.5
CONDITION_NUMBER = UPPER_HESSIAN_BOUND / LOWER_HESSIAN_BOUND

DIMENSION = 2

#%%
#Constants related to MChain

INITIAL_DISTRIBUTION = multivariate_normal(cov= 1/UPPER_HESSIAN_BOUND * np.eye(DIMENSION))
error_tol = 0.2

ERROR_TOLS = [0.2,0.1,1.0]


#%%
#Empirical draws
EMPIRICAL_SIZE = 500000

def draw_empirical_samples(empirical_size,
                           gaussian_center,
                           dimension):
    
    
    bernoulli_samples = np.random.rand(empirical_size,1) < 0.5
    standard_normal_samples = multivariate_normal.rvs(mean = np.zeros((dimension)),
                                                      size = empirical_size)
    empirical_samples = ((standard_normal_samples - gaussian_center)*bernoulli_samples + 
                    (standard_normal_samples + gaussian_center)*(1-bernoulli_samples))
    
    return empirical_samples

EMPIRICAL_SAMPLES = draw_empirical_samples(EMPIRICAL_SIZE, GAUSSIAN_CENTER,
                                           DIMENSION)

#%%
#Extend to nd
def discrete_tv_error(normalized_hist1, normalized_hist2):
    
    normalizing_const = np.sum(normalized_hist1)
    
    return (np.sum(np.abs(normalized_hist1 - normalized_hist2)) 
            / normalizing_const)
#%%
HISTOGRAM_RANGE = [[-5,5],[-5,5]]
NO_OF_BINS = 30
HISTOGRAM_LEVELS = np.array([0.004, 0.009, 0.025, 0.05, 0.08])
#HISTOGRAM_LEVELS = np.array([0.004, 0.009, 0.025, 0.05, 0.1])

EMPIRICAL_HISTOGRAM,_,_ = np.histogram2d(EMPIRICAL_SAMPLES[:,0],
                                         EMPIRICAL_SAMPLES[:,1],
                                         bins=NO_OF_BINS,range = HISTOGRAM_RANGE,
                                         density = True)

#%%
#2D Discretisation error
def extent_of_discretisation_error(runs = 100, histogram_range = HISTOGRAM_RANGE,
                         no_of_bins = NO_OF_BINS):
    
    errors = []
    
    for i in range(runs):
        
        sample_1 = draw_empirical_samples(EMPIRICAL_SIZE,
                                          GAUSSIAN_CENTER,
                                          DIMENSION)
        sample_2 = draw_empirical_samples(EMPIRICAL_SIZE,
                                          GAUSSIAN_CENTER,
                                          DIMENSION)
        hist_1,_,_ = np.histogram2d(sample_1[:,0],
                                    sample_1[:,1],
                                    bins=no_of_bins,range = histogram_range,
                                    density = True)
        hist_2,_,_ = np.histogram2d(sample_2[:,0],
                                    sample_2[:,1],
                                    bins=no_of_bins,range = histogram_range,
                                    density = True)

        #normalization_cost = np.sum(hist_1)
        normalization_cost = 1
        
        errors.append(discrete_tv_error(hist_1/normalization_cost, 
                                        hist_2 / normalization_cost))
        
    return min(errors),max(errors)


MIN_DISCRETISATION_ERROR, MAX_DISCRETISATION_ERROR = extent_of_discretisation_error()

print(MIN_DISCRETISATION_ERROR, MAX_DISCRETISATION_ERROR)
#%%
PROJECTION_AXIS_ONE = np.array([1,1]).reshape((-1,1)) / np.sqrt(2)
PROJECTION_AXIS_TWO = np.array([1,-1]).reshape((-1,1)) / np.sqrt(2)

PROJECTION_MATRIX_ONE = PROJECTION_AXIS_ONE @ (PROJECTION_AXIS_ONE.T)
PROJECTION_MATRIX_TWO = PROJECTION_AXIS_TWO @ (PROJECTION_AXIS_TWO.T)

NO_OF_ONE_DIM_BINS = 30
ONE_DIM_HISTOGRAM_RANGE = [-5,5]

AXIS_ONE_PROJECTED_SAMPLES = (EMPIRICAL_SAMPLES @ PROJECTION_MATRIX_ONE 
                              / (PROJECTION_AXIS_ONE.T))[:,0]
AXIS_ONE_EMPIRICAL_HISTOGRAM,_ = np.histogram(AXIS_ONE_PROJECTED_SAMPLES,
                                              bins = NO_OF_ONE_DIM_BINS,
                                              range = ONE_DIM_HISTOGRAM_RANGE,
                                              density = True)

AXIS_TWO_PROJECTED_SAMPLES = (EMPIRICAL_SAMPLES @ PROJECTION_MATRIX_TWO
                              / (PROJECTION_AXIS_TWO.T))[:,0]
AXIS_TWO_EMPIRICAL_HISTOGRAM,_ = np.histogram(AXIS_TWO_PROJECTED_SAMPLES,
                                              bins = NO_OF_ONE_DIM_BINS,
                                              range = ONE_DIM_HISTOGRAM_RANGE,
                                              density = True)




#%%
#Draw samples from algorithm at a fixed iteration

def draw_samples_at_fixed_iteration(algo_final_state,
                               potential, grad_potential, target_density,
                               initial_distribution, step_size,
                               sampled_iteration, no_of_sampled_points,
                               save=False, name = '',
                               verbose = 1, part = 100):
    final_states = []
    fraction = 1
    
    if verbose>0:
            print(f"Total iterations to run {sampled_iteration}")

    for i in range(no_of_sampled_points):
        
        final_states.append(algo_final_state(initial_distribution, 
                                             step_size, sampled_iteration,
                                             grad_potential, potential,
                                             target_density))
        if verbose>0:
            if i == int(no_of_sampled_points * fraction/part):
                print(f"{fraction}/{part} is done!")
                fraction += 1
        
    final_states = np.array(final_states)
    
    if save:
        np.save(name, final_states)
    
    return final_states 

#%%
#Draw samples from algorithm at a fixed time

def draw_samples_at_fixed_time(algo_final_state,
                               potential, grad_potential, target_density,
                               initial_distribution, step_size,
                               sampled_time, no_of_sampled_points,
                               save=False, name = '',
                               verbose = 1,part = 100):
    
    max_iterations = int(sampled_time/step_size)

    return draw_samples_at_fixed_iteration(algo_final_state,
                               potential, grad_potential, target_density,
                               initial_distribution, step_size,
                               max_iterations, no_of_sampled_points,
                               save, name,
                               verbose, part)


#%%
sampled_time = 500
no_of_sampled_points = 70000

step_size = ULA.set_step_size(error_tol, DIMENSION,
                              CONDITION_NUMBER,
                              UPPER_HESSIAN_BOUND, LOWER_HESSIAN_BOUND)

final_states = draw_samples_at_fixed_time(ULA.final_state,
                                          POTENTIAL, GRAD_POTENTIAL, 
                                          TARGET_DENSITY,
                                          INITIAL_DISTRIBUTION, step_size,
                                          sampled_time,no_of_sampled_points,
                                          save = False, name = 'ULA-time = 500-points = 70000.npy')
final_states.shape

#%%
max_iterations = 100
no_of_sampled_points = 500

step_size = 0.5

final_states = draw_samples_at_fixed_time(HMC.final_state,
                                          POTENTIAL, GRAD_POTENTIAL, 
                                          TARGET_DENSITY,
                                          INITIAL_DISTRIBUTION, step_size,
                                          max_iterations,no_of_sampled_points,
                                          save = True, name = 'HMC')
final_states.shape

#%%
#Histograms 


def plot_contour_histogram(ax, points,
                           histogram_range, no_of_bins, levels,
                           colors = 'red', label = None, display_levels = True):
    
    '''Computes the 2D histogram and plots it on ax for points a 2D array, with
    diff cols representing x and y axes'''

    density,xbins,ybins = np.histogram2d(points[:,0],points[:,1],
                                    bins=no_of_bins,range = histogram_range,
                                    density=True)

    print(np.sum(density))
    
    Xbins,Ybins = np.meshgrid(xbins[:-1], ybins[:-1])
    CS = ax.contour(Xbins, Ybins, density, levels, colors = colors)
    
    if label != None:
        CS.collections[0].set_label(label)
    
    if display_levels:
        ax.clabel(CS, inline=True, fontsize=12)
        
    


#%%
#algo_samples = np.load('ULA-time = 500-points = 50000.npy')
algo_samples = final_states

fig, ax = plt.subplots(figsize=(10,10))


plot_contour_histogram(ax,EMPIRICAL_SAMPLES,
                       HISTOGRAM_RANGE,30,HISTOGRAM_LEVELS,
                       colors = 'blue', label = 'TARGET')

plot_contour_histogram(ax,algo_samples,
                       HISTOGRAM_RANGE,30,HISTOGRAM_LEVELS,
                       colors = 'red', label = 'ULA',
                       display_levels=False)

#ax.set_title(f'ULA - Contour Plot- h = {step_size}, sampled time = {sampled_time}, points = {no_of_sampled_points}')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
#plt.savefig(f'ULA - Contour Plot')
plt.show()




#%%
def draw_paths_until_fixed_iteration(algo_states,
                                potential, grad_potential, target_density,
                                initial_distribution, step_size,
                                max_iterations, no_of_sampled_paths=1,
                                save=False, name = '',
                                verbose = 1, part = 100):
    
    sampled_paths = []
    fraction = 1
    #sampled_paths.shape =  (no_of_paths,no_of_iterations,dimension)

    for path_no in range(no_of_sampled_paths):
        
        path = algo_states(initial_distribution, 
                           step_size, max_iterations,
                           grad_potential, potential,
                           target_density)
        sampled_paths.append(path)
        
        if verbose>0:
            if i == int(no_of_sampled_paths * fraction/part):
                print(f"{fraction}/{part} is done!")
                fraction += 1
        
        
    
    sampled_paths = np.array(sampled_paths)
    
    if save:
        np.save(name, sampled_paths)
    
    
    
    return sampled_paths


def draw_paths_until_fixed_time(algo_states,
                                potential, grad_potential, target_density,
                                initial_distribution, step_size,
                                max_time, no_of_sampled_paths=1,
                                save=False, name = '',
                                verbose = 1, part = 100):
    
    max_iterations = int(max_time/step_size)
    return draw_paths_until_fixed_iteration(algo_states,
                                            potential, grad_potential, target_density,
                                            initial_distribution, step_size,
                                            max_iterations, no_of_sampled_paths,
                                            save, name,
                                            verbose,part)


#%%

max_iterations = 600
no_of_sampled_paths = 10

step_size = ULA.set_step_size(0.2, DIMENSION,
                         CONDITION_NUMBER, 
                         UPPER_HESSIAN_BOUND, LOWER_HESSIAN_BOUND)

sampled_paths = draw_paths_until_fixed_iteration(ULA.states,
                                            POTENTIAL, GRAD_POTENTIAL, 
                                            TARGET_DENSITY,
                                            INITIAL_DISTRIBUTION, step_size,
                                            max_iterations,no_of_sampled_paths,
                                            name = 'ULA')


#%%
def plot_curves(ax, curve_list, time,
                labels, xlabel, ylabel,
                single_curve = False, same_times = True, start_time = 0,
                title=None, xticks =  [], yticks = [],
                colors = [], show_grid = False,
                xlog = False, ylog = False,
                save_plot=False, plot_name = None,
                show=True):
    
    if single_curve:
        curve_list = [curve_list]
        labels = [labels]
        
    if same_times:
        
        for i,curve in enumerate(curve_list):
            
            if len(colors) > 0:
                
                ax.plot(time, curve, colors[i], label=labels[i])
            else:
                
                ax.plot(time, curve, label=labels[i])
    
    else:
        
        for i,curve in enumerate(curve_list):
            
            if len(colors) > 0:
                
                ax.plot(time[i], curve, colors[i], label=labels[i])
            else:
                
                ax.plot(time[i], curve, label=labels[i])
                
                
    
    if title != None:
        ax.set_title(title)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if xlog:
        ax.set_xscale('log')
        f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.1e' % x))
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(g))
        #ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%.e'))
    if ylog:
        ax.set_yscale('log')
        f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.1e' % x))
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(g))
        #ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%.e'))
    
    if len(xticks) > 0:
        ax.set_xticks(xticks)
    
    if len(yticks) > 0:
        ax.set_yticks(yticks)
    
    if show_grid:
        ax.grid(True)
    
    ax.legend()
    
    if save_plot:
        plt.savefig(plot_name)
        
    if show:
        plt.show()
    
    return ax


#%%
'''#Can be optimized to improve runtime
# d dimensions
#BUGS ARE THERE HERE!!
def mean_TV_errors_with_time(sampled_paths,
                        bins, histogram_range,
                        empirical_histogram=None, empirical_samples = None,
                        save = False, name = ''):
    
    #sampled_paths.shape =  (no_of_paths,no_of_iterations,dimension)
    no_of_paths = sampled_paths.shape[0]
    max_iterations = sampled_paths.shape[1]
    
    mean_TV_errors = []
    

    if empirical_histogram == None:
        empirical_histogram,edges = np.histogramdd(empirical_samples,
                                         bins=bins,range = histogram_range,
                                         density = True)
    
    normalizing_const = np.sum(empirical_histogram)
    
    #empirical_density = empirical_histogram / normalizing_const
    empirical_density = empirical_histogram / normalizing_const
    
    #print(normalizing_const)
    
    for iteration in range(max_iterations):
        
        avg_TV_error_until_iteration = 0
            
        for path_number in range(no_of_paths):
            
            sampled_histogram,_ = np.histogramdd(sampled_paths[path_number,:iteration+1,:],
                                                     bins=bins, range = histogram_range, 
                                                     density = True)
           
            #sampled_density = sampled_histogram / normalizing_const
            sampled_density = sampled_histogram / normalizing_const
           
            avg_TV_error_until_iteration = ((avg_TV_error_until_iteration*path_number 
                             + discrete_tv_error(empirical_density,sampled_density))
                            / (path_number+1))
        
        mean_TV_errors.append(avg_TV_error_until_iteration)
    
    mean_TV_errors = np.array(mean_TV_errors)
        
    
    if save:
        np.save(name,mean_TV_errors)
    
    
    return mean_TV_errors
'''
#%%
def mean_TV_errors_at_time_2D(sampled_paths,
                        bins,histogram_range,
                        empirical_histogram = np.array([]),
                        empirical_samples = None,
                        save = False,name = ''):
    
    #sampled_paths.shape =  (no_of_paths, no_of_iterations, dimension)
    no_of_paths = sampled_paths.shape[0]
    max_iterations = sampled_paths.shape[1]
    

    if empirical_histogram.size == 0:
        empirical_histogram,_,_ = np.histogram2d(empirical_samples[:,0],
                                                   empirical_samples[:,1],
                                                   bins=bins,range = histogram_range,
                                                   density = True)
    
    #normalizing_const = np.sum(empirical_histogram)

    
    #empirical_density = empirical_histogram / normalizing_const
    empirical_density = empirical_histogram 
    avg_TV_error = 0
            
    for path_number in range(no_of_paths):
        
        sampled_histogram,_,_ = np.histogram2d(sampled_paths[path_number,:max_iterations+1,0],
                                             sampled_paths[path_number,:max_iterations+1,1],
                                             bins=bins, range = histogram_range, 
                                             density = True)
       
        sampled_density = sampled_histogram 
       
        avg_TV_error = ((avg_TV_error * path_number 
                        + discrete_tv_error(empirical_density,sampled_density))
                        / (path_number + 1))

    return avg_TV_error
        




def mean_TV_errors_with_time_2D(sampled_paths,step_size,
                                bins,histogram_range,
                                start_time = 0,
                                burn_in_time = 0,
                                empirical_histogram = np.array([]),
                                empirical_samples = None,
                                save = False,name = ''):
    
    #sampled_paths.shape =  (no_of_paths, no_of_iterations, dimension)
    no_of_paths = sampled_paths.shape[0]
    max_iterations = sampled_paths.shape[1]
    
    start_iteration = int(start_time/step_size)
    burn_in_iteration = int(burn_in_time/step_size)
    
    mean_TV_errors = []
    

    if empirical_histogram.size == 0:
        empirical_histogram,_,_ = np.histogram2d(empirical_samples[:,0],
                                                   empirical_samples[:,1],
                                                   bins=bins,range = histogram_range,
                                                   density = True)
    
    #normalizing_const = np.sum(empirical_histogram)
    normalizing_const = 1 
    
    #empirical_density = empirical_histogram / normalizing_const
    empirical_density = empirical_histogram / normalizing_const
    
    for iteration in range(start_iteration,max_iterations):
        
        avg_TV_error_until_iteration = 0
            
        for path_number in range(no_of_paths):
            
            sampled_histogram,_,_ = np.histogram2d(
                                                 sampled_paths[path_number,burn_in_iteration:iteration+1,0],
                                                 sampled_paths[path_number,burn_in_iteration:iteration+1,1],
                                                 bins=bins, range = histogram_range, 
                                                 density = True)
           
            sampled_density = sampled_histogram / normalizing_const
           
            avg_TV_error_until_iteration = ((avg_TV_error_until_iteration*path_number 
                             + discrete_tv_error(empirical_density,sampled_density))
                            / (path_number+1))
        
        mean_TV_errors.append(avg_TV_error_until_iteration)
        
    mean_TV_errors = np.array(mean_TV_errors)
        
    
    if save:
        np.save(name,mean_TV_errors)
    
    
    return mean_TV_errors





def projected_mean_TV_errors_with_time(projection_axis,
                                sampled_paths,
                                bins,histogram_range,
                                projected_empirical_histogram = np.array([]),
                                empirical_samples = None,
                                save = False,name = ''):
    
    no_of_paths = sampled_paths.shape[0]
    max_iterations = sampled_paths.shape[1]
    
    projection_matrix = projection_axis @ (projection_axis.T)

    if projected_empirical_histogram.size == 0:
        
        projected_samples_1D = ((empirical_samples @ projection_matrix)
                                / (projection_axis).T)[:,0]
        
        projected_empirical_histogram,_ = np.histogram(projected_samples_1D,
                                                     bins=bins,range=histogram_range,
                                                     density=True)
        
    
    #normalizing_const = np.sum(projected_empirical_histogram)
    normalizing_const = 1
    
    #empirical_density = projected_empirical_histogram / normalizing_const
    empirical_density = projected_empirical_histogram / normalizing_const
    
    mean_TV_errors = []

    for iteration in range(max_iterations):
        
        avg_TV_error_until_iteration = 0
            
        for path_number in range(no_of_paths):
            
            projected_samples_1D = ((sampled_paths[path_number,:iteration+1,:] @ projection_matrix)
                                / (projection_axis).T)[:,0]
            
            
            sampled_histogram,_ = np.histogram(projected_samples_1D,
                                           bins=bins, range = histogram_range, 
                                           density = True)
       
            sampled_density = sampled_histogram / normalizing_const
           
            avg_TV_error_until_iteration = ((avg_TV_error_until_iteration*path_number 
                             + discrete_tv_error(empirical_density,sampled_density))
                            / (path_number+1))
        
        mean_TV_errors.append(avg_TV_error_until_iteration)
        
    mean_TV_errors = np.array(mean_TV_errors)
    
    
    

    return mean_TV_errors
    



def projected_mean_TV_errors_at_time(projection_axis,
                                        sampled_paths,
                                        bins,histogram_range,
                                        projected_empirical_histogram = np.array([]),
                                        empirical_samples = None,
                                        save = False,name = ''):
    
    no_of_paths = sampled_paths.shape[0]
    #max_iterations = sampled_paths.shape[1]
    
    projection_matrix = projection_axis @ (projection_axis.T)

    if projected_empirical_histogram.size == 0:
        
        projected_samples_1D = ((empirical_samples @ projection_matrix)
                                / (projection_axis).T)[:,0]
        
        projected_empirical_histogram,_ = np.histogram(projected_samples_1D,
                                                     bins=bins,range=histogram_range,
                                                     density=True)
        
    
    #normalizing_const = np.sum(projected_empirical_histogram)
    normalizing_const = 1
    
    #empirical_density = projected_empirical_histogram / normalizing_const
    empirical_density = projected_empirical_histogram / normalizing_const
    
    avg_TV_error = 0
            
    for path_number in range(no_of_paths):
        
        projected_samples_1D = ((sampled_paths[path_number,:,:] @ projection_matrix)
                                / (projection_axis).T)[:,0]
        
        sampled_histogram,_ = np.histogram(projected_samples_1D,
                                           bins=bins, range = histogram_range, 
                                           density = True)
       
        sampled_density = sampled_histogram / normalizing_const
       
        avg_TV_error = ((avg_TV_error * path_number 
                        + discrete_tv_error(empirical_density,sampled_density))
                        / (path_number + 1))

    return avg_TV_error
    
    

#%%
max_time = 200
no_of_sampled_paths = 5


names = ['ULA','ULA_small','ULA_big']
times = []

STEP_SIZES = [0.05,0.1,0.2]

for i,step_size in enumerate(STEP_SIZES):
    
    max_iterations = int(max_time/step_size)
        
    times.append(np.linspace(0,max_time,max_iterations+1))
    
    sampled_paths = draw_paths_until_fixed_time(ULA.states,
                                                POTENTIAL, GRAD_POTENTIAL,
                                                TARGET_DENSITY,
                                                INITIAL_DISTRIBUTION, step_size,
                                                max_time, no_of_sampled_paths,
                                                save=True,
                                                name = names[i])
    

                                                          
#%%
tv_error_curves = []
for i,name in enumerate(names):
    tv_error_curves.append(mean_TV_errors_with_time_2D(np.load(name + '.npy'),
                           0.01, NO_OF_BINS,
                           HISTOGRAM_RANGE,
                           empirical_histogram = EMPIRICAL_HISTOGRAM,
                           save = False))

#%%

#y_values = [5e-3,6e-3,7e-3,8e-3,9e-3,1e-2,2e-2]
fig,ax = plt.subplots()
ax = plot_curves(ax, tv_error_curves, times,
                ['h=0.05','h=0.1','h=0.2'], 'time', 'TV_errors',
                same_times = False,
                show_grid = True,ylog = True,
                title = 'ULA - TV Errors VS Time',
                save_plot = False, plot_name = 'ULA - TV Errors VS Time')


#ax.plot(range(),MIN_DISCRETISATION_ERROR * np.ones((800)))
#ax.plot(range(800),MAX_DISCRETISATION_ERROR * np.ones((800)))
plt.show()

#%%
print(tv_error_curves[0][-1],tv_error_curves[1][-1],tv_error_curves[2][-1])
#print(MAX_DISCRETISATION_ERROR)

#%%
#plt.yscale('log')
plt.yscale('log')
plt.yticks([2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2,2e-2])
plt.grid(True,axis='both')
#plt.yaxis().set_major_formatter(ticker.ScalarFormatter())
plt.plot(times,tv_error_curves[-1])
#%%
def draw_sample_paths_until_fixed_time(algo_states,
                                       potential, grad_potential, target_density,
                                       initial_distribution, step_sizes,
                                       max_time, no_of_sampled_paths=1,
                                       save=False, name = ''):
    
    if isinstance(step_sizes, (int,float)):
        step_sizes = np.array([step_sizes])
    
    sampled_paths = []
    
    for step_size in step_sizes:
        
        max_iterations = int(max_time/step_size)
        sampled_paths_for_a_step_size = []
        
        #sampled_paths_for_a_step_size.shape =  (no_of_paths,no_of_iterations,dimension)
    
        for path_no in range(no_of_sampled_paths):
            
            path = algo_states(initial_distribution, 
                               step_size, max_iterations,
                               grad_potential, potential,
                               target_density)
            sampled_paths_for_a_step_size.append(path)
        
        sampled_paths.append(np.array(sampled_paths_for_a_step_size))

    
    if len(step_sizes) == 1:
        step_sizes = step_sizes[0]
        sampled_paths = sampled_paths[0]
        
    if save:
        np.save(name, sampled_paths)
    
    return sampled_paths

#%%
MAX_TIMES = [2000]
no_of_sampled_paths = 10

#STEP_SIZES = np.array([0.01,0.02,0.04,0.08])
STEP_SIZES = np.linspace(0.001,0.1,10)


for j,max_time in enumerate(MAX_TIMES):
        
    sampled_paths = draw_sample_paths_until_fixed_time(ULA.states,
                                                       grad_potential = GRAD_POTENTIAL,
                                                       potential = POTENTIAL,
                                                       target_density = TARGET_DENSITY,
                                                       initial_distribution = INITIAL_DISTRIBUTION, 
                                                       step_sizes = STEP_SIZES,
                                                       max_time = max_time,
                                                       no_of_sampled_paths = no_of_sampled_paths,
                                                       save=False)
    
    

#%%
mean_TV_errors = []
for i,step_size in enumerate(STEP_SIZES):
    mean_error = (mean_TV_errors_at_time_2D(sampled_paths[i],
                                            NO_OF_BINS,
                                            HISTOGRAM_RANGE,
                                            empirical_histogram =
                                            EMPIRICAL_HISTOGRAM))
    
    
    mean_TV_errors.append(mean_error)
 
    
    
#%%
'''    
mean_TV_errors = []
for i,step_size in enumerate(STEP_SIZES):
    mean_error_axis_1 = (projected_mean_TV_errors_at_time(PROJECTION_AXIS_ONE,
                                                          sampled_paths[i],
                                                          NO_OF_ONE_DIM_BINS,
                                                          ONE_DIM_HISTOGRAM_RANGE,
                                                          projected_empirical_histogram =
                                                          AXIS_ONE_EMPIRICAL_HISTOGRAM))
    
    mean_error_axis_2 = (projected_mean_TV_errors_at_time(PROJECTION_AXIS_TWO,
                                                          sampled_paths[i],
                                                          NO_OF_ONE_DIM_BINS,
                                                          ONE_DIM_HISTOGRAM_RANGE,
                                                          projected_empirical_histogram =
                                                          AXIS_TWO_EMPIRICAL_HISTOGRAM))
    
    
    mean_TV_errors.append((mean_error_axis_1 + mean_error_axis_2)/2)
    
'''
#%%
    
fig,ax = plt.subplots()
ax.plot(STEP_SIZES,mean_TV_errors,'og',label=f'T = {MAX_TIMES[0]}')

m,b = np.polyfit(STEP_SIZES,mean_TV_errors,1)
ax.plot(STEP_SIZES,m*STEP_SIZES + b,label=f'Least Square Line',color = 'red')

f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.4e' % x))
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(g))
#ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(g))
#ax.set_xticks(np.linspace(0.01,0.3,10))
ax.grid(True)
ax.set_xlabel('h')
ax.set_ylabel('Proj_TV_error')
ax.set_title("Discretisation bias - ULA")

ax.legend()
plt.show()    
