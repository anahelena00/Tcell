#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
import os
from datetime import datetime

#%%

file_name = '20240102-01-50_100000_1e5_5d.npz'

npzfile = np.load(file_name)

keys = npzfile.files
data_dict = {key: npzfile[key] for key in keys}
locals().update(data_dict)

T = data_dict['T'] 
eps = data_dict['eps'] 
density = data_dict['density']
size_mini = data_dict['size_mini']
ind_equi = data_dict['ind_equi'],
E_mean = data_dict['E_mean']
E_var = data_dict['E_var']
num_runs = data_dict['num_runs']

ind_equi = ind_equi[0].item()
    
# %%

def stirling(x):
    res = x*np.log(x)-x
    return res

def multiplicity(MinLatticeSize, density):
    d_max = max(density)
    n = MinLatticeSize**2*d_max
    N = n/density
    multiplicity = stirling(N)-stirling(n)-stirling(N-n)
    
    num_particles = N
    return num_particles, multiplicity

def the_physics(T, E_mean, E_var, rho, MinLatticeSize):

    N, M = multiplicity(MinLatticeSize, rho)

    S_ref = np.log(M)
    
    Cv_gradient = np.zeros([len(T), len(rho)])
    Cv_variance = np.zeros([len(T), len(rho)])
    Sgrad = np.zeros([len(T), len(rho)])
    Svar = np.zeros([len(T), len(rho)])
    Fgrad = np.zeros([len(T), len(rho)])
    Fvar = np.zeros([len(T), len(rho)])
    for d in range(len(rho)):
        Cv_gradient[:,d] = np.gradient(E_mean[:,d],T)
        for i in range(len(T)):
            Cv_variance[i,d] = E_var[i,d]/T[i]**2
            
        Sgrad[:,d] = S_ref[d] - np.cumsum(Cv_gradient[:,d])
        Svar[:,d] = S_ref[d] - np.cumsum(Cv_variance[:,d])
        Fgrad[:,d] = E_mean[:,d] - T*Sgrad[:,d]
        Fvar[:,d] = E_mean[:,d] - T*Svar[:,d]
    
    return Cv_gradient, Cv_variance, Sgrad, Svar, Fgrad, Fvar

Cv_grad, Cv_var, Sgrad, Svar, Fgrad, Fvar =the_physics(T, E_mean, E_var, density, size_mini)
#%%
def lots_of_plots(density, datax, xlabel, datay, ylabel):

    num_cols = 1
    num_rows = len(density) # Ceiling division to calculate the number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 5*num_rows))

    for i, ax in enumerate(axes.flat):
        if i < len(density):  # To handle the case where len(d) is not a multiple of num_cols
            ax.plot(datax, datay[:, i], 'o')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('{} (d={:.2f})'.format(ylabel, density[i]))

    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()
    plt.show()

def plot_for_all_d(density, datax, xlabel, datay1, datay2, ylabel):

    num_cols = 1
    num_rows = len(density) # Ceiling division to calculate the number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 5*num_rows))

    for i, ax in enumerate(axes.flat):
        if i < len(density):  # To handle the case where len(d) is not a multiple of num_cols
            ax.plot(datax, datay1[:, i], 'o')
            ax.plot(datax, datay2[:, i], 'o')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('{} (d={:.2f})'.format(ylabel, density[i]))

    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()
    plt.show()

#%%

lots_of_plots(density, T, 'T', E_mean, 'U')

num_cols = 1
num_rows = len(density) # Ceiling division to calculate the number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))

for i, ax in enumerate(axes.flat):
    if i < len(density):  # To handle the case where len(d) is not a multiple of num_cols
        ax.plot(T,Cv_var[:,i],'o', label = 'from variance' )
        ax.plot(T,Cv_grad[:,i],'o', label = 'from gradient' )
        ax.set_xlabel('T')
        ax.set_ylabel('{} (d={:.2f})'.format('Cv', density[i]))
        ax.legend()

# Adjust layout to prevent overlap of subplots
plt.tight_layout()
plt.show()

#%%
fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 5*num_rows))

for i, ax in enumerate(axes.flat):
    if i < len(density):  # To handle the case where len(d) is not a multiple of num_cols
        ax.plot(T,Svar[:,i],'o', label = 'from variance' )
        ax.plot(T,Sgrad[:,i],'o', label = 'from gradient' )
        ax.set_xlabel('T')
        ax.set_ylabel('{} (d={:.2f})'.format('S', density[i]))
        ax.legend()

# Adjust layout to prevent overlap of subplots
plt.tight_layout()
plt.show()

#%%

fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 5*num_rows))

for i, ax in enumerate(axes.flat):
    if i < len(density):  # To handle the case where len(d) is not a multiple of num_cols
        ax.plot(T,Fvar[:,i],'o', label = 'from variance' )
        ax.plot(T,Fgrad[:,i],'o', label = 'from gradient' )
        ax.set_xlabel('T')
        ax.set_ylabel('{} (d={:.2f})'.format('F', density[i]))
        ax.legend()

# %%

fit_grad_list = []
fit_var_list = []
xp = np.linspace(min(T), max(T), 500)
for run_idx in range(Cv_grad.shape[1]):
    
    fit_grad = np.poly1d(np.polyfit(T, Cv_grad[:,run_idx], 8))
    fit_grad_list.append(fit_grad)
    fit_var = np.poly1d(np.polyfit(T, Cv_var[:,run_idx],  8))
    fit_var_list.append(fit_var)
   # print(fit_var)
    
    plt.figure()
    plt.plot(T, Cv_grad[:,run_idx], 'o', color = 'orange', label = 'From gradient')
    plt.plot(xp, fit_grad(xp), color='C3', label = 'gradient fit')
    plt.plot(T, Cv_var[:,run_idx], '.', markersize=10, color = 'darkblue', label = 'From variance')
    plt.plot(xp, fit_var(xp), color = 'darkturquoise', label = 'variance fit')
    plt.xlabel('Temperature')
    plt.ylabel('$C_{V}$')
    plt.title(f'density={density[run_idx]}, Iterations per T: {num_runs}')
    plt.legend()
    plt.show()


#fit_grad_list = np.array(fit_grad_list)
#fit_var_list = np.array(fit_var_list)   
# %%

# Finding temperatue for Cv max
    
def Tpeak(T, Cv):
    idx = np.argmax(Cv)
    Tp = T[idx]
    return Tp

#%%
Tp_grad = np.zeros(len(density))
Tp_var = np.zeros(len(density))
for i in range(len(density)):
    Tp_grad[i] = Tpeak(xp, fit_grad_list[i](xp))
    Tp_var[i] = Tpeak(xp, fit_var_list[i](xp))

plt. figure()
plt.plot(density, Tp_grad,'o', label = 'from gradient')
plt.plot(density, Tp_var,'o', label = 'from variance')
plt.xlabel('density')
plt.ylabel('T')
plt.ylim([0,1])
plt.legend()
plt.show()
# %%
