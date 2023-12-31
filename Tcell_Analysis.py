#%%

import numpy as np
import math as math
import matplotlib.pyplot as plt

#%% 
file_name = '20240103-04-54_1e5_BT_size100.npz'
#file_name = '20231231-19-47_7e4_T033_B1_s50.npz'
#file_name = '20231231-20-11_7e4_T033_B1_s70.npz'
#file_name = '20231231-20-39_7e4_T033_B1_s100.npz'
npzfile = np.load(file_name)

keys = npzfile.files
data_dict = {key: npzfile[key] for key in keys}
locals().update(data_dict)
#print('keys:',keys)    
T = data_dict['T'] 
eps = data_dict['eps'] 
muT = data_dict['muT']
muB = data_dict['muB']
T_num_in = data_dict['T_num_in']
T_num = data_dict['T_num']
B_num_in = data_dict['B_num_in']
B_num_history = data_dict['B_num_history']
size = data_dict['size']
ind_equi = data_dict['ind_equi'],
E_mean = data_dict['E_mean']
E_var = data_dict['E_var']
num_runs = data_dict['num_runs']

ind_equi = ind_equi[0].item()


#%%

def stirling(x):
    res = x*np.log(x)-x
    return res

def ln_multiplicity(x):
    if x > 20:
        res = x*np.log(x)-x     # stirlings approximation
    else:
        res = np.log(math.factorial(x))
    return res

def S_reference(size, B_num, T_num):
    N = size**2
    N_0 = N - B_num - T_num # N_0=0 if full then omit from Omega
    S_ref = np.zeros(len(B_num), dtype = int)
    for i in range(len(S_ref)):
        S_ref[i] = ln_multiplicity(N) - ln_multiplicity(T_num[i]) - ln_multiplicity(B_num[i]) - ln_multiplicity(N_0[i])
    return S_ref

def the_physics(T, E_mean, E_var, B_num, muB, T_num, muT, size):
     
    S_ref = S_reference(size, B_num, T_num)   
#    print(S_ref)
    G_T = muT*T_num
   # G_B = muB*B_num[-1]
    G_B = muB*B_num
    
    x = (B_num+T_num)/size**2    
    G = (1-x)*G_T + x*G_B - T*(x*np.log(x)+(1-x)*np.log(1-x))
    
    Cv_variance = E_var/T**2
    Cv_gradient = np.gradient(E_mean, T)        
    Sgrad = S_ref - np.cumsum(Cv_gradient)
    Svar = S_ref - np.cumsum(Cv_variance)
    F = E_mean - T*Sgrad
     
    return Cv_gradient, Cv_variance, Sgrad, Svar, F, G

#%% 
B_num = np.zeros(len(T), dtype = int)
for i in range(len(T)):
    B_num[i] = int(np.mean(B_num_history[i][ind_equi:]))
B_rho = B_num/size**2

# B density
plt.figure()
plt.plot(T, B_rho, '.')
plt.xlabel('T')
plt.ylabel('density')
#plt.yticks([0,1])
plt.show()

#%%  
Cv_grad, Cv_var, Sgrad, Svar, F, G = the_physics(T, E_mean, E_var, B_num, muB, T_num, muT, size) 

#%%
# FIGURES 


# Mean Energy
plt.figure()
plt.plot(T, E_mean,'o')
plt.xlabel('Temperature')
plt.ylabel('Mean Energy')
#plt.title(f'Mean Energy, T_num: {Tcell_num}, B_num: {Bacteria_num}')
#plt.savefig(f'Runs_data/{datetime_str}U_mean.pdf', format = 'pdf')
plt.show()

# Gibbs Energy
plt.figure()
plt.plot(T,G,'o')
plt.xlabel('Temperature')
plt.ylabel('Gibbs energy')
plt.show()

# Helmholtz Free Energy
plt.figure()
plt.plot(T,F,'o')
plt.xlabel('Temperature')
plt.ylabel('F')
#plt.title('Helmholtz Free Energy')
plt.show()

# Entropy
plt.figure()
plt.plot(T,Sgrad,'o', label = 'from variance' )
plt.plot(T,Svar,'o', label = 'from gradient' )
plt.xlabel('Temperature')
plt.ylabel('Entropy')
#plt.title('Entropy')
plt.show()

# Heat capacity   
plt.figure()
plt.plot(T,Cv_grad,'o', color = 'orange', label = 'From gradient')
plt.plot(T,Cv_var, '.', markersize = '10', color = 'darkblue', label = 'From variance')
plt.xlabel('Temperature')
plt.ylabel('$C_{V}$')
plt.title(f'Heat capacity')
plt.legend()
#plt.savefig(f'Runs_data/{datetime_str}Cv.png')
plt.show()


# %%

# Fit Cv and find Cv Max

fit_grad_list = []
fit_var_list = []
T_for_fit = np.linspace(min(T), max(T), 500)
for run_idx in range(Cv_grad.shape[1]):
    
    fit_grad = np.poly1d(np.polyfit(T, Cv_grad[:,run_idx], 3))
    fit_grad_list.append(fit_grad)
    fit_var = np.poly1d(np.polyfit(T, Cv_var[:,run_idx],  3))
    fit_var_list.append(fit_var)
    
    plt.figure()
    plt.plot(T, Cv_grad[:,run_idx], 'o', color = 'orange', label = 'From gradient')
    plt.plot(T_for_fit, fit_grad(T_for_fit), color='C3', label = 'gradient fit')
    plt.plot(T, Cv_var[:,run_idx], '.', markersize=10, color = 'darkblue', label = 'From variance')
    plt.plot(T_for_fit, fit_var(T_for_fit), color = 'darkturquoise', label = 'variance fit')
    plt.xlabel('Temperature')
    plt.ylabel('$C_{V}$')
    plt.title(f'density={B_rho[run_idx]}, Iterations per T: {num_runs}')
    plt.legend()
    plt.show()

# %%
