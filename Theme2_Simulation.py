#%%

import numpy as np
import random
import matplotlib.pyplot as plt
import numba
from datetime import datetime

#%%

def create_lattice(size, P_num):
    lattice = np.zeros([size,size], dtype=int)
    rng = np.random.default_rng()
    lat_idx = np.argwhere(lattice == 0)
    #print('lattice indices before shuffling',lat_idx)
    rng.shuffle(lat_idx)
    #print('lattice indices after shuffling',lat_idx)
    lat_idx = lat_idx.T

    # number of holes
    hole_num = int(size**2 - P_num)
    #print(hole_num)
    assert hole_num + P_num == size**2
    #coor_0 = lat_idx[:,T_num+B_num:T_num+B_num+hole_num]
    coor_0 = lat_idx[:,P_num:P_num+hole_num]
    coor_T = lat_idx[:,0:P_num]

    # Assign values to coordinates
    lattice[coor_T[0], coor_T[1]] = 1
    #lattice[coor_B[0], coor_B[1]] = 2

    pos0 = -1*np.ones([2,size**2], dtype = int)
    pos0[:,0:hole_num] = coor_0

    # allocate (with -1's) b_coords
    posT = -1*np.ones([2,size**2],dtype=int)
    posT[:,0:P_num] = coor_T

    density = P_num/size**2

    return lattice, posT, pos0, density

#%%

def position_random(pos): 

    if np.any(pos < 0):
       num_coords = np.where(np.any(pos < 0, axis=0))[0][0] - 1
       if num_coords == 0:
           col = num_coords
           #print('1',col)
        # if there are more than 1 set of coordinates
       if num_coords > 0: 
           col = np.random.randint(pos[:,0:num_coords].shape[1])
           #print('2',col)
    else:
        col = np.random.randint(pos.shape[1])
        #print('3',col)
    p = pos[:,col]
    return p, col

#%%

# find energy a particle has or would have at specified position

def energy(lattice,p,eps):
    s = lattice.shape[0]-1
    i = p[0]
    j = p[1]
    
    if i==0:
        up = lattice[s,j]   
    else:    
        up = lattice[i-1,j]

    if i == s:
        down = lattice[0,j]      
    else:
        down = lattice[i+1,j]

    if j == 0:
        left = lattice[i,s]        
    else:
        left = lattice[i,j-1]     

    if j == s:
        right = lattice[i,0]       
    else:
        right = lattice[i,j+1]
        
    E = -eps * 1 *(up+down+left+right) # 1 added because their /is/ or /hypothetically is/ a particle
    return E

#%%

# total energy of lattice
def lattice_energy(lattice, eps):
    
    # interaction energy
    E_interaction = 0
    rows, cols = lattice.shape
    for i in range(rows):
        for j in range(cols):
            #val = int(lattice[i, j])
            E_interaction += energy(lattice, (i, j), eps)
    E_interaction = E_interaction / 2
    
    return E_interaction

#%%

def evaluate_particle_move(lattice, pos1, pos0, T, E_total, eps):
   
    p1, col1 = position_random(pos1)
    
   # ID_in = lattice[p1[0], p1[1]]
    #assert ID_in == 1, f"ID_in not 1 but {ID_in}"
    Ein = energy(lattice, p1, eps)
    
    p0, col0 = position_random(pos0)
    #assert lattice[p0[0],p0[1]] == 0 , f"Found position is not empty"
    # seeing what energy would be for particle if it moved the the chosen empty location
    lattice[p1[0], p1[1]] = 0 # temporarily moving object so not to be seen as neighbor by itself
    Efin = energy(lattice, p0, eps)
    Ediff = Efin - Ein

    if Ediff < 0 :
        move = True
    
    else:
        probability = np.exp(-Ediff/T)
        if random.random() < probability: # random.random gives between 0 and 1. Hence higher prob -> more move
            move = True 
        else:
            move = False
        
    if move: 
        pos1 = pos1.copy()
        pos1[:,col1] = [p0[0], p0[1]]
        lattice[p0[0], p0[1]] = 1
        # update arrays containing coordinates of 0's and 1's
        pos0 = pos0.copy()
        pos0[:,col0] = [p1[0], p1[1]]
        #assert lattice[pos1[0][col1],pos1[1][col1]] == 1, f"not 1 but: {lattice[pos1[0][col1],pos1[1][col1]]}"
        #assert lattice[pos0[0][col0],pos0[1][col0]] == 0, f"not 0 but: {lattice[pos0[0][col0],pos0[1][col0]]}"
        E_total = E_total + Ediff
        
    else:
        lattice[p0[0], p0[1]] = 0
        lattice[p1[0], p1[1]] = 1

    return lattice, pos1, pos0, E_total

#%% 

def gridprint(lattice):
    lattice_length = len(lattice)
    cmap = plt.cm.colors.ListedColormap(['white', 'blue'])
    plt.imshow(lattice, cmap=cmap, extent=[0, lattice_length, 0, lattice_length], vmin=0, vmax=2)
    plt.colorbar(ticks=[0, 1], label="Legend")
   # plt.title("")
    #plt.grid(True, linewidth=0.5, color='black')
    plt.show()


#%%
def monte_carlo(T, eps, lattice_length, number_of_particles, num_runs, num_lattices_to_store=None):
    
    # initiate lattice and find 0's and 1's
    lattice, pos1, pos0, density = create_lattice(lattice_length, number_of_particles)    

    E_lattice = lattice_energy(lattice, eps)

  #  if num_lattices_to_store is not None:
   #     # Calculate the step size to evenly space the lattices
    #    step_size = max(1, num_runs // (num_lattices_to_store - 2))
#
 #       # Create a list to store the indices of the selected lattices
  #      selected_indices = [0] + [i for i in range(step_size, num_runs - step_size, step_size)] + [num_runs - 1]


 #   lattice_history = {}
    E_history = {}

    for j in range(len(T)):

        lattice_history_for_T = []
        E_history_for_T = []

        for i in range(0,num_runs): # change to from one and append initial E and lattice to outisde
            
            E_history_for_T.append(E_lattice)
            
#            if num_lattices_to_store is not None:
 #           
  #              if i in selected_indices:
   #                 lattice_history_for_T.append(lattice)


            lattice, pos1, pos0, E_lattice = evaluate_particle_move(lattice, pos1, pos0, T[j], E_lattice, eps)


        E_history[j] = E_history_for_T
    #    if num_lattices_to_store is not None:
     #       lattice_history[j] = lattice_history_for_T

    return E_history, density
 
#%%


@numba.jit
def process_loop(T, eps, density_approx, number_of_particles, num_runs):    

    E_history_tempo_list=[]
    density_tempo_list=[]
    for i, d in enumerate(density_approx):
        size = int(np.sqrt(number_of_particles / d))
        E_history, density = monte_carlo(T, eps, size, number_of_particles, num_runs, num_lattices_to_store=None)
        E_history_tempo_list.append(E_history)
        density_tempo_list.append(density)
        
    # Sort density_approx and get the indices that would sort it
    sorted_indices = np.argsort(density_approx)

    # Rearrange E_history_list and density_list
    E_history_list = [E_history_tempo_list[i] for i in sorted_indices]
    density_list = [density_tempo_list[i] for i in sorted_indices]

    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime('%Y%m%d-%H-%M_') 
    run_name = f'{datetime_str}{num_runs}'
        
    return E_history_list, density_list, run_name

# Process the loop using the optimized function
E_history_list, density_list = process_loop(T, eps, density_approx, number_of_particles, num_runs)

#%%


num_runs = 100_000
T_interval1 = np.arange(3, 2, 0.2)
T_interval2 = np.arange(2, 1, -0.05)
T_interval3 = np.arange(1, 0.01, -0.001)
T = np.concatenate((T_interval1, T_interval2, T_interval3))
T = np.arange(3,0.01,-0.1)
eps = 1        
         
size_mini = 50  

#densityUp = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
#densityDown = [0.5, 0.45, 0.4, 0.3, 0.2, 0.1]
#density_approx = densityDown + densityUp        # approximate because densities are found in monte carlo 
#density_approx = [0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]
density_approx = [0.95, 0.8, 0.6, 0.7, 0.5]
P_num = int(max(density_approx) * size_mini**2)

#%%
E_history, density, run_name = process_loop(T, eps, density_approx, P_num, num_runs)

#%%
def mean_energy(T, density, E_history, ind_equilibrium):

    E_mean = np.zeros([len(T),len(density)])
    E_variance = np.zeros([len(T),len(density)])

    for j in range(len(density)):
        E = E_history[j]
        for i in range(len(T)):
            E_mean[i][j] = np.mean(E[i][ind_equilibrium:-1])
            E_variance[i][j] = np.var(E[i][ind_equilibrium:-1])
            
    return E_mean, E_variance
 
ind_equi = int(0.8*num_runs)
E_mean, E_var = mean_energy(T, density, E_history, ind_equi)

#%%

# SAVE DATA 

file_spec = '1e5_5d'   # extra info for filename. Customize
file_name = f'{run_name}_{file_spec}.npz'

np.savez(file_name, 
         T = T,
         eps = eps,
         density = density,
         size_mini = size_mini,
         ind_equi = ind_equi,
         E_mean = E_mean,
         E_var = E_var,
         num_runs = num_runs,        
        )
print(file_name)

# %%
