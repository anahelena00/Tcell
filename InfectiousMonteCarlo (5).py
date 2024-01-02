
#%%


import numpy as np
import random
import matplotlib.pyplot as plt
import numba
import time
import os
from datetime import datetime
from mpmath import mp


#%%

def create_lattice(size, T_num, B_num):
    lattice = np.zeros([size,size], dtype=int)
    rng = np.random.default_rng()
    lat_idx = np.argwhere(lattice == 0)
    #print('lattice indices before shuffling',lat_idx)
    rng.shuffle(lat_idx)
    #print('lattice indices after shuffling',lat_idx)
    lat_idx = lat_idx.T

    # number of holes
    hole_num = int(size**2 - T_num - B_num)
    #print(hole_num)
    assert hole_num + T_num + B_num == size**2
    coor_0 = lat_idx[:,T_num+B_num:T_num+B_num+hole_num]
    coor_B = lat_idx[:,T_num:T_num+B_num]
    coor_T = lat_idx[:,0:T_num]

    # Assign values to coordinates
    lattice[coor_T[0], coor_T[1]] = 1
    lattice[coor_B[0], coor_B[1]] = 2

    pos0 = -1*np.ones([2,size**2], dtype = int)
    pos0[:,0:hole_num] = coor_0
    # 
    posT = -1*np.ones([2,size**2], dtype = int)
    posT[:,0:T_num] = coor_T

    # allocate (with -1's) b_coords
    posB = -1*np.ones([2,size**2],dtype=int)
    posB[:,0:B_num] = coor_B
    ''' 
    rng = np.random.default_rng()
    lat_idx = np.argwhere(lattice == 0)
    #print('lattice indices before shuffling',lat_idx)
    rng.shuffle(lat_idx)
    #print('lattice indices after shuffling',lat_idx)
    lat_idx = lat_idx.T

    # number of holes
    hole_num = int(size**2 - T_num - B_num)
    #print(hole_num)
    assert hole_num + T_num + B_num == size**2
    coor_0 = lat_idx[:,T_num+B_num:T_num+B_num+hole_num]
    coor_B = lat_idx[:,T_num:T_num+B_num]
    coor_T = lat_idx[:,0:T_num]

    # Assign values to coordinates
    lattice[coor_T[0], coor_T[1]] = 1
    lattice[coor_B[0], coor_B[1]] = 2

    pos0 = -1*np.ones([2,size**2], dtype = int)
    pos0[:,0:hole_num] = coor_0
    # 
    posT = coor_T
    # allocate (with -1's) b_coords
    posB = -1*np.ones([2,size**2],dtype=int)
    posB[:,0:B_num] = coor_B
    '''
    return lattice, posT, posB, pos0

#%%
# showing lattices as they evolve 

def lattice_plots(lattice_history, selected_indices):

    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red'])

    for i in range(len(selected_indices)):

        # Create a plot
        plt.imshow(lattice_history[i], cmap=cmap, extent=[0, size, 0, size])
        plt.colorbar(ticks=[0, 1, 2], label="Legend")
        plt.title("Lattice with T's (Blue) and B's (Red)")
        plt.show()
        

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
def create_unmix_lattice(size, num_1s, num_2s): #TAKES A NUMBER OF BS AND TS AND PUTS THEM IN LEFT/RIGHT

    lattice = np.zeros([size, size], dtype=int)

    # Calculate the number of holes
    hole_num = size**2 - num_2s - num_1s
    
    rows= size
    # Place type 2 particles at the top
    #lattice[:num_2s, :] = 2
    for i in range(rows):
        if  num_2s>0:
            lattice[:,0:num_2s][i] =2
            num_2s = num_2s -size
    #print(lattice)
    # Place type 1 particles at the bottom
    #lattice[-num_1s:, :] = 1
    for i in range(rows):
        if  num_1s>0:
            lattice[:,0:num_1s][size-i-1] =1
            num_1s = num_1s -size
    #lattice[:,0:num_1s][-1] =1
    #print(lattice)
    # Randomly select positions for type 0 particles without replacement
    coor_0 = np.argwhere(lattice == 0)
    #print(coor_0)

    # Find coordinates of each type of particle
    coor_2 = np.argwhere(lattice == 2)
    coor_0 = np.argwhere(lattice == 0)
    coor_1 = np.argwhere(lattice == 1)
    
    
    # Initialize position arrays
    pos2 = coor_2.T
    pos0 = coor_0.T
    pos1 = coor_1.T

    return lattice, pos2, pos1, pos0

#%%

def energy(lattice, ID_in, pos_hypo, interaction_matrix):
    s = lattice.shape[0]-1
    i = pos_hypo[0] # x coordinate
    j = pos_hypo[1] # y coordinate
    
    if i==0:
        up = lattice[s,j]   
    else:    
        up = lattice[i-1,j]
    up = int(up)

    if i == s:
        down = lattice[0,j]      
    else:
        down = lattice[i+1,j]
    down = int(down)

    if j == 0:
        left = lattice[i,s]        
    else:
        left = lattice[i,j-1]    
    left = int(left)

    if j == s:
        right = lattice[i,0]       
    else:
        right = lattice[i,j+1]
    right = int(right)
    
    
    E = -(interaction_matrix[ID_in, up] + interaction_matrix[ID_in, down] + interaction_matrix[ID_in, left] + 
          interaction_matrix[ID_in, right])
    return E


#%%

def lattice_energy(lattice, eps, muT, muB):
    
    # interaction energy
    E_interaction = 0
    rows, cols = lattice.shape
    for i in range(rows):
        for j in range(cols):
            val = int(lattice[i, j])
            E_interaction += energy(lattice, val, (i, j), eps)
    E_interaction = E_interaction / 2
    
    # chemical energy
    E_chemical = muT*np.sum(lattice==1) + muB*np.sum(lattice==2)
    
    # total energy
    E_total = E_interaction + E_chemical
    
    return E_total

#%%

def evaluate_particle_addB(lattice, pos2, pos0, T, E_total, eps, muB, B_num):
    
    p0, colb = position_random(pos0) #pick a hole to put bacteria
    
  #  assert lattice[p0[0],p0[1]] == 0
    #ID_in= lattice[pb[0], pb[1]] #change it in the lattice
    ID_B = 2
    Efin = energy(lattice, ID_B, p0, eps) + muB # energy from interaction and chemical if placed
    #ID_in = lattice[pb[0], pb[1]]
    ID_empty = 0
    Ein = energy(lattice,ID_empty, p0, eps) #evaluate neighbouring energy before
    
    #print("Efin , Ein", Efin, Ein)
    Ediff = Efin - Ein 
    #print("Ediff ", Ediff)

    if Ediff < 0 :
        add = True
        #print('addB: Ediff<0')
    
    else:
        probability = np.exp(-Ediff/T)
        #print('p_B:', probability)
        if random.random() < probability: # random.random gives between 0 and 1. Hence higher prob -> more move
            add = True
#            print('addB:', E_total, Ediff) 
        else:
            add = False
        
    if add: 
        lattice[p0[0], p0[1]] = 2
        #print("before",pos2, pb)
        #pos0 = np.delete(pos0, colb, axis=1)
        #print('before', pos0, colb)
        pos0 = pos0.copy()
        pos0[:,colb:-1] = pos0[:,colb+1:]
        #print('after', pos0)
        first_negative_column = np.where(np.any(pos2 < 0, axis=0))[0][0]
        pos2 = pos2.copy()
        pos2[:,first_negative_column] = p0
        #print(pos2)
        E_total = E_total + Ediff
        B_num = B_num + 1
        #print("Ediff:", Ediff)
        
    else:
        lattice[p0[0], p0[1]] = 0
    #print(E_total, Ediff)
    return lattice, pos2, pos0, E_total, B_num

#%%

def evaluate_particle_addT(lattice, pos1, pos0, T, E_total, eps, muT, T_num):
    
    p0, colT = position_random(pos0) #pick a hole to put bacteria
    
  #  assert lattice[p0[0],p0[1]] == 0
    #ID_in= lattice[pb[0], pb[1]] #change it in the lattice
    ID_B = 1
    Efin = energy(lattice, ID_B, p0, eps) + muT # energy from interaction and chemical if placed
    #ID_in = lattice[pb[0], pb[1]]
    ID_empty = 0
    Ein = energy(lattice,ID_empty, p0, eps) #evaluate neighbouring energy before
    
    #print("Efin , Ein", Efin, Ein)
    Ediff = Efin - Ein 
    #print("Ediff ", Ediff)

    if Ediff < 0 :
        add = True
        #print('addB: Ediff<0')
    
    else:
        probability = np.exp(-Ediff/T)
        #print('p_B:', probability)
        if random.random() < probability: # random.random gives between 0 and 1. Hence higher prob -> more move
            add = True
#            print('addB:', E_total, Ediff) 
        else:
            add = False
        
    if add: 
        lattice[p0[0], p0[1]] = 2
        #print("before",pos2, pb)
        #pos0 = np.delete(pos0, colb, axis=1)
        #print('before', pos0, colb)
        pos0 = pos0.copy()
        pos0[:,colT:-1] = pos0[:,colT+1:]
        #print('after', pos0)
        first_negative_column = np.where(np.any(pos1 < 0, axis=0))[0][0]
        pos1 = pos1.copy()
        pos1[:,first_negative_column] = p0
        #print(pos2)
        E_total = E_total + Ediff
        T_num = T_num + 1
        #print("Ediff:", Ediff)
        
    else:
        lattice[p0[0], p0[1]] = 0
    #print(E_total, Ediff)
    return lattice, pos1, pos0, E_total, T_num

#%%

def evaluate_particle_removeB(lattice, posB, pos0, T, E_total, eps, muB, B_num):

    pB, colB = position_random(posB)
    ID_B = lattice[pB[0], pB[1]]
   # assert ID_B == 2, f"ID_B not 2 but {ID_B}"
    Ein = energy(lattice, ID_B, pB, eps)
    Efin = energy(lattice, 0, pB, eps) - muB  # energy if particle is removed 
    Ediff = Efin - Ein

    if Ediff < 0:
        remove = True
#        print('removeB: Ediff<0')
    else: 
        probability = np.exp(-Ediff/T)
        if random.random() < probability: # random.random gives between 0 and 1. Hence higher prob -> more move
            remove = True 
            #print('removeB:', E_total, Ediff)
        else:
            remove = False
    if remove:
        # remove B from posB        
        posB = posB.copy()
        posB[:,colB:-1] = posB[:,colB+1:]

        # add 0 on old Bs coordinates
        first_negative_column = np.where(np.any(pos0 < 0, axis=0))[0][0]
        pos0 = pos0.copy()
        pos0[:,first_negative_column] = pB
        lattice[pB[0], pB[1]] = 0
        E_total = E_total + Ediff
        B_num = B_num -1
    else:
        pass

    return lattice, posB, pos0, E_total, B_num

#%%
#CHANGE BELLOW!
def evaluate_particle_removeT(lattice, posT, pos0, T, E_total, eps, muT, T_num):

    pT, colT = position_random(posT)
    ID_B = lattice[pT[0], pT[1]]
   # assert ID_B == 2, f"ID_B not 2 but {ID_B}"
    Ein = energy(lattice, ID_B, pT, eps)
    Efin = energy(lattice, 0, pT, eps) - muT  # energy if particle is removed 
    Ediff = Efin - Ein

    if Ediff < 0:
        remove = True
#        print('removeB: Ediff<0')
    else: 
        probability = np.exp(-Ediff/T)
        if random.random() < probability: # random.random gives between 0 and 1. Hence higher prob -> more move
            remove = True 
            #print('removeB:', E_total, Ediff)
        else:
            remove = False
    if remove:
        # remove B from posB        
        posT = posT.copy()
        posT[:,colT:-1] = posT[:,colT+1:]

        # add 0 on old Bs coordinates
        first_negative_column = np.where(np.any(pos0 < 0, axis=0))[0][0]
        pos0 = pos0.copy()
        pos0[:,first_negative_column] = pT
        lattice[pT[0], pT[1]] = 0
        E_total = E_total + Ediff
        T_num = T_num -1
    else:
        pass

    return lattice, posT, pos0, E_total, T_num

#%%

def evaluate_particle_moveT(lattice, pos1, pos0, T, E_total, eps):
   
    p1, col1 = position_random(pos1)
    
    ID_in = lattice[p1[0], p1[1]]
    #assert ID_in == 1, f"ID_in not 1 but {ID_in}"
    Ein = energy(lattice,ID_in, p1, eps)
    
    p0, col0 = position_random(pos0)
    #assert lattice[p0[0],p0[1]] == 0 , f"Found position is not empty"
    # seeing what energy would be for particle if it moved the the chosen empty location
    lattice[p1[0], p1[1]] = 0 # temporarily moving object so not to be seen as neighbor by itself
    Efin = energy(lattice, ID_in, p0, eps)
    #print("Efin , Ein", Efin, Ein)
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
        #print("Ediff:", Ediff)
        
    else:
        lattice[p0[0], p0[1]] = 0
        lattice[p1[0], p1[1]] = 1
    #print(E_total, Ediff)
    return lattice, pos1, pos0, E_total

#%%

def gridprint(lattice):
    lattice_length = len(lattice)
    cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red'])
    plt.imshow(lattice, cmap=cmap, extent=[0, lattice_length, 0, lattice_length], vmin=0, vmax=2)
    plt.colorbar(ticks=[0, 1, 2], label="Legend")
    plt.title("Lattice with T's (Blue) and B's (Red)")
    #plt.grid(True, linewidth=0.5, color='black')
    plt.show()


#%%
def monte_carlo(Temp, eps, lattice_length, T_num_in, B_num_in, muT, muB, num_runs, num_lattices_to_store=None):
    
    E_history = {}
    T_num = np.zeros(len(Temp), dtype=int)
    B_num_history = []
    #E_history_unmixed ={}
    for ind, t in enumerate(Temp):
        E_history_for_Temp = []
        E_history_for_Temp_unmixed =[]
        lattice, pos1, pos2, pos0 = create_lattice(lattice_length, T_num_in, B_num_in)
        E_lattice = lattice_energy(lattice, eps, muT, muB)
        B_num_for_Temp = np.zeros(num_runs, dtype = int)
        B_num = B_num_in
        #gridprint(lattice)
        for i in range(0,num_runs): # change to from one and append initial E and lattice to outisde
            E_history_for_Temp.append(E_lattice)
            #print(i)
            if np.all(pos0 < 0): # if no holes -> only attempt remove B and T
                lattice, pos2, pos0, E_lattice, B_num = evaluate_particle_removeB(
                    lattice, pos2, pos0, t, E_lattice, eps, muB, B_num)
                lattice, pos1, pos0, E_lattice, T_num = evaluate_particle_removeT(lattice, pos1, pos0, t, E_lattice, eps, muT, T_num)
                #print("Lattice is full at iteration:", i)
                
            elif np.all(pos2 < 0): # if no B's -> only attempt move, add and remove T and add B 
                lattice, pos1, pos0, E_lattice = evaluate_particle_moveT(
                                              lattice, pos1, pos0, t, E_lattice, eps)
                lattice, pos2, pos0, E_lattice, B_num = evaluate_particle_addB(
                                                lattice, pos2, pos0, t, E_lattice, eps, muB, B_num)
                selected_functionT = random.choice([evaluate_particle_addT, evaluate_particle_removeT])
                lattice, pos1, pos0, E_lattice, T_num = selected_functionT(lattice, pos1, pos0, t, E_lattice, eps, muT, T_num)
                #assert B_num + T_num_in <= lattice_length**2, f"To many B's. {B_num}"
                
            elif np.all(pos1 < 0): # if no T's -> only attempt move and add T and add/remove B
                lattice, pos1, pos0, E_lattice = evaluate_particle_moveT(
                                              lattice, pos1, pos0, t, E_lattice, eps)
                lattice, pos1, pos0, E_lattice, T_num = evaluate_particle_addT(lattice, pos1, pos0, t, E_lattice, eps, muT, T_num)
                selected_functionB = random.choice([evaluate_particle_addB, evaluate_particle_removeB])
                lattice, pos2, pos0, E_lattice, B_num = selected_functionB(lattice, pos2, pos0, t, E_lattice, eps, muB, B_num)
                #assert B_num + T_num_in <= lattice_length**2, f"To many B's. {B_num}"  
                                 
            else: # attempt all 
                lattice, pos1, pos0, E_lattice = evaluate_particle_moveT(
                                              lattice, pos1, pos0, t, E_lattice, eps)
                selected_functionB = random.choice([evaluate_particle_addB, evaluate_particle_removeB])
                lattice, pos2, pos0, E_lattice, B_num = selected_functionB(lattice, pos2, pos0, t, E_lattice, eps, muB, B_num)
                selected_functionT = random.choice([evaluate_particle_addT, evaluate_particle_removeT])
                lattice, pos1, pos0, E_lattice, T_num = selected_functionT(lattice, pos1, pos0, t, E_lattice, eps, muT, T_num)
                #lattice, pos2, pos0, E_lattice, B_num = evaluate_particle_addB(lattice, pos2, pos0, t, E_lattice, eps, muB, B_num)
                #assert B_num + T_num_in <= lattice_length**2, f"To many B's. {B_num}" 
            B_num_for_Temp[i] = B_num
        #gridprint(lattice)
        B_num_history.append(B_num_for_Temp)
        #Sum of the energy of 2 lattices: one with just T-cells and the other with just bacteria
        U_unmix = lattice_energy(create_unmix_lattice(lattice_length, 0,B_num)[0], eps, muT, muB) +lattice_energy(create_unmix_lattice(lattice_length, T_num_in,0)[0], eps, muT, muB)
        E_history_for_Temp_unmixed.append( U_unmix)
         
      #  pos2t.append(pos2.shape[1])
            
            #pos0t.append(pos0)
            #pos1t.append(pos1)
            
        T_num[ind] = np.sum((pos1 != -1).all(axis=0))
        #B_num[ind] = np.sum((pos2 != -1).all(axis=0))
        #Tcell.append(pos1.shape[1])   

        E_history[t] = E_history_for_Temp.copy()
  
    # Unique name for data file 
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime('%Y%m%d-%H-%M')    
    run_name = f'{datetime_str}'
    print(E_history_for_Temp_unmixed) 
    return lattice, E_history, B_num_history, T_num, run_name #, pos0_hist, pos1_hist, pos2_hist

#%%


# the interaction matrix can be used to decide how many bacteria are able to multiply. 
# if surrounded by T cells -> no division
# the body is modelled by an N by N lattice

num_runs = 1000
T_interval1 = np.arange(40, 10, -1)
T_interval2 = np.arange(10, 3, -0.5)
T_interval3 = np.arange(3, 0.2, -0.2)
T = np.concatenate((T_interval1, T_interval2, T_interval3))
#T = np.arange(20, 0.1, -1)
#T = np.arange(1,0.01,-0.5)
#T = np.arange(.1,.01,-0.1) ##Test
size = 50

#T_num_in = int(size**2/2)    # number of initial T-cells
#T_num_in  = int(size**2/3)
T_num_in  = int(size*2/3)
B_num_in = int(1)
muT, muB = -1, -2.1

BB_int = -1     # interaction energy between bacterias
TT_int = -1      # interaction energy between T-cells
BT_int = 4     # interaction energy between bacteria and T-cells
interaction_matrix = np.array([
    [0, 0, 0],
    [0, TT_int, BT_int],
    [0, BT_int, BB_int]
])
#%% This is to get U_unmixed for Gibbs energy
#print(create_unmix_lattice(size, T_num_in, B_num_in)[0][0], size, T_num_in, B_num_in)
#%%
lattice, E_history, B_num_history, T_num, run_name = monte_carlo(T, interaction_matrix, size, T_num_in, B_num_in, muT, muB, num_runs, num_lattices_to_store=None)

#%%
def B_num_plot(B_num_history, T, size):
    yMin = 0
    yMax = size**2
    num_cols = 2
    num_rows = len(T) // num_cols
    _, axes = plt.subplots(num_rows, num_cols, figsize=(8, 2*len(T)))
    axes = axes.flatten()
    for i in range(len(T)):
        ax = axes[i]
        T_formatted = f'{T[i]:.2f}'
        ax.plot(np.arange(0, num_runs), B_num_history[i], '.', markersize = '0.5')
        ax.set_yticks(np.arange(yMin, yMax, step = int(yMax/10)))
        ax.set_ylabel(f'N_B, T = {T_formatted}')
    plt.tight_layout()
    plt.show()
B_num_plot(B_num_history, T, size)


#%%

def mean_energy(T, E_history, ind_equilibrium):
    
    E = E_history
    E_mean = np.zeros([len(T)])
    E_variance = np.zeros([len(T)])

    for ind,t in enumerate(T):
        E_mean[ind] = np.mean(E[t][ind_equilibrium:])
        E_variance[ind] = np.var(E[t][ind_equilibrium:])

    return E_mean, E_variance

ind_equi = int((0.7)*num_runs) # index where equilibrium is assumed. 
E_mean, E_var = mean_energy(T, E_history, ind_equi)

#%%
def E_history_plot(E_history, T, num_runs):

    yMin = min(E_history)-1
    yMax = max(E_history)

    E_keys = list(E_history.keys())
    # Create subplots
    _, axes = plt.subplots(len(T), 1, figsize=(8, 2*len(T)))
    for i in range(len(T)):
        ax = axes[i]
        T_formatted = f'{T[i]:.2f}'
        ax.plot(np.arange(0, num_runs), E_history[E_keys[i]], '.', markersize = '0.1')
        ax.set_yticks(np.arange(yMin, yMax, step = 20))
        ax.set_ylabel(f'Energy (T = {T_formatted})')
        #ax.set_xlabel('Iteration')
        #ax.set_title()

    plt.show()


E_history_plot(E_history, T, num_runs)
#%%

plt.figure()
plt.plot(T,E_mean,'o')
plt.xlabel('T') 
plt.ylabel('U')
plt.title(f'Size: {size}, Runs: {num_runs}')
plt.show()
#lattice_plots(lattice, np.arange(0,100,5))


#%%
# SAVE DATA 

file_spec = '1e5_T033_B1'   # extra info for filename. Customize
file_name = f'{run_name}_{file_spec}.npz'

np.savez(file_name, 
         T = T,
         eps = interaction_matrix,
         muT = muT,
         muB = muB,
         T_num_in = T_num_in,
         T_num = T_num,
         B_num_in = B_num_in,
         B_num_history = B_num_history,
         size = size,
         ind_equi = ind_equi,
         E_mean = E_mean,
         E_var = E_var,
         num_runs = num_runs,        
        )
print(file_name)

# %%