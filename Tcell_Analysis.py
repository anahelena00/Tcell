#%%

import numpy as np
import math as math
import matplotlib.pyplot as plt

#%% 

file_name = '20231228-22-36_1e5_detailedBalance.npz'
npzfile = np.load(file_name)

keys = npzfile.files
for key in keys:
    locals()[key] = npzfile[key]
print('keys:',keys)    
# keys: ['T', 'eps', 'muT', 'muB', 'T_num', 'B_num', 'size', 'E_mean', 'E_variance', 'num_runs']

E_var = E_variance
#T_num = Tcell_num


#%%

"""
def stirling(x):
    res = x*np.log(x)-x
    return res

def multiplicity2(size, B_num):
    N = size**2
    multiplicity = stirling(N)-stirling(B_num)-stirling(N-B_num)
    return multiplicity
"""

def multiplicity(size, B_num, T_num):
    N = size**2
    N_0 = N - B_num - T_num
    Omega = np.zeros(len(B_num))
    for i in range(len(Omega)):
        Omega[i] = math.factorial(N)/(math.factorial(N_0[i]) 
            * math.factorial(B_num[i]) * math.factorial(T_num[i]))
    return Omega
    

def the_physics(T, E_mean, E_var, M, B_num, muB, T_num, muT, size):
     
    S_ref = np.log(M)   
    G_T = muT*T_num
   # G_B = muB*B_num[-1]
    G_B = muB*B_num
    
    x = (B_num+T_num)/size**2    
    G = (1-x)*G_T + x*G_B - T*(x*np.log(x)+(1-x)*np.log(1-x))
    
    Cv_variance = E_var/T**2
    Cv_gradient = np.gradient(E_mean,T)        
    Sgrad = S_ref - np.cumsum(Cv_gradient)
    Svar = S_ref - np.cumsum(Cv_variance)
    F = E_mean - T*Sgrad
     
    return Cv_gradient, Cv_variance, Sgrad, Svar, F, G

#%% 
M = multiplicity(size, B_num, T_num)
Cv_grad, Cv_var, Sgrad, Svar, F, G =the_physics(T, E_mean, E_var, M, B_num, muB, T_num, muT,size) 

#%%
# FIGURES 

# Mean Energy
plt.figure()
plt.plot(T,E_mean,'o')
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
