
#%%

import numpy as np
import random
import matplotlib.pyplot as plt
import numba
import time
import os
from datetime import datetime
from mpmath import mp
import matplotlib.colors as mcolors
import random

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
    posT = coor_T
    # allocate (with -1's) b_coords
    posB = -1*np.ones([2,size**2],dtype=int)
    posB[:,0:B_num] = coor_B

    return lattice, posT, posB, pos0
"""    
def create_lattice(size, T_num, B_num):
    lattice = np.zeros([size,size], dtype=int)
    rng = np.random.default_rng()
    lat_idx = np.argwhere(lattice == 0)
    #print('lattice indices before shuffling',lat_idx)
    rng.shuffle(lat_idx)
    #print('lattice indices after shuffling',lat_idx)
    lat_idx = lat_idx.T
    t_coords = lat_idx[:,0:T_num]
    #print('t: \n',t_coords)
    b_coords = lat_idx[:,T_num:T_num+B_num]
    #print('b: \n',b_coords)

    # number of holes
    hole_num = int(size**2 - T_num - B_num)
    #print(hole_num)
    empty_coords = -1*np.ones([2,size**2], dtype = int)
   # print(empty_coords, empty_coords.shape[1])
    empty_coords[:,0:hole_num] = lat_idx[:,T_num+B_num:]
    #print(empty_coords, empty_coords.shape[1])
    #print('O: \n',empty_cords)

    # allocate (with -1's) b_coords
    b_alloc = -1*np.ones([2,size**2],dtype=int)
    b_alloc[:,0:B_num] = b_coords

    # Assign values to coordinates
    lattice[t_coords[0], t_coords[1]] = 1
    lattice[b_coords[0], b_coords[1]] = 2

    return lattice, t_coords, b_alloc, empty_coords

"""

"""
def create_lattice(size, T_num, B_num):
    lattice = np.zeros([size,size], dtype=int)
    rng = np.random.default_rng()
    lat_idx = np.argwhere(lattice == 0)
    #print('lattice indices before shuffling',lat_idx)
    rng.shuffle(lat_idx)
    #print('lattice indices after shuffling',lat_idx)
    lat_idx = lat_idx.T
    t_coords = lat_idx[:,0:T_num]
    #print('t: \n',t_coords)
    b_coords = lat_idx[:,T_num:T_num+B_num]
    #print('b: \n',b_coords)
    empty_coords = lat_idx[:,T_num+B_num:]
    #print('O: \n',empty_cords)
    
    # Assign values to coordinates
    lattice[t_coords[0], t_coords[1]] = 1
    lattice[b_coords[0], b_coords[1]] = 2

    # allocate (with -1's) b_coords
    b_alloc = -1*np.ones([2,size**2],dtype=int)
    b_alloc[:,0:B_num] = b_coords

    return lattice, t_coords, b_alloc, empty_coords
"""

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

"""
def position_random(pos): 
    col = np.random.randint(pos.shape[1]) 
    p = pos[:,col]
    return p, col
"""

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

def evaluate_particle_addB(lattice, pos2, pos0, T, E_total, eps, muB):
    
    pb, colb = position_random(pos0) #pick a hole to put bacteria
    
    assert lattice[pb[0],pb[1]] == 0
    #ID_in= lattice[pb[0], pb[1]] #change it in the lattice
    ID_B = 2
    Efin = energy(lattice, ID_B, pb, eps) + muB # energy from interaction and chemical if placed
    #ID_in = lattice[pb[0], pb[1]]
    ID_empty = 0
    Ein = energy(lattice,ID_empty, pb, eps) #evaluate neighbouring energy before
    
    #print("Efin , Ein", Efin, Ein)
    Ediff = Efin - Ein 
    #print("Ediff ", Ediff)

    if Ediff < 0 :
        add = True
    
    else:
        probability = np.exp(-Ediff/T)
        #print('p_B:', probability)
        if random.random() < probability: # random.random gives between 0 and 1. Hence higher prob -> more move
            add = True 
        else:
            add = False
        
    if add: 
        lattice[pb[0], pb[1]] = 2
        #print("before",pos2, pb)
        #pos0 = np.delete(pos0, colb, axis=1)
        #print('before', pos0, colb)
        pos0[:,colb:-1] = pos0[:,colb+1:]
        #print('after', pos0)
        first_negative_column = np.where(np.any(pos2 < 0, axis=0))[0][0]
        pos2[:,first_negative_column] = pb
        #print(pos2)
        E_total = E_total + Ediff
        #print("Ediff:", Ediff)
        
    else:
        lattice[pb[0], pb[1]] = 0
    #print(E_total, Ediff)
    return lattice, pos2, pos0, E_total

"""
def evaluate_particle_addB(lattice, pos2, pos1, pos0, T, E_total, eps, muT, muB):
    #print("pos1before",pos1.shape)
    pb, colb = position_random(pos0) #pick a hole to put bacteria
    
    #ID_in= lattice[pb[0], pb[1]] #change it in the lattice
    ID_B = 2
    Efin = energy(lattice, ID_B, pb, eps) + muB # energy from interaction and chemical if placed
    #ID_in = lattice[pb[0], pb[1]]
    ID_empty = 0
    Ein = energy(lattice,ID_empty, pb, eps) #evaluate neighbouring energy before
    
    #print("Efin , Ein", Efin, Ein)
    Ediff = Efin - Ein 
    #print("Ediff ", Ediff)

    if Ediff < 0 :
        add = True
    
    else:
        #mp.dps = 128
        #probability = np.float128(np.exp(-(Ediff -muB*pos2.shape[1] -muT*pos1.shape[1])/T) )
        #probability = mp.exp(-(Ediff - muB * pos2.shape[1] - muT * pos1.shape[1]) / T)
        probability = np.exp(-Ediff/T)
        #print('p_B:', probability)
        if random.random() < probability: # random.random gives between 0 and 1. Hence higher prob -> more move
            add = True 
        else:
            add = False
        
    if add: 
        lattice[pb[0], pb[1]] = 2
        #print("before",pos2, pb)
        pos0 = np.delete(pos0, colb, axis=1)
       # pos2 = np.hstack((pos2, np.array([pb]).reshape(-1, 1)))
        first_negative_column = np.where(np.any(pos2 < 0, axis=0))[0][0]
        pos2[:,first_negative_column] = pb
        #print(pos2)
        E_total = E_total + Ediff
        #print("Ediff:", Ediff)
        
    else:
        lattice[pb[0], pb[1]] = 0
    #print(E_total, Ediff)
    return lattice, pos2, pos0, E_total
"""

#%%


def evaluate_particle_removeB(lattice, pos2, pos0, T, E_total, eps, muB):
    
    p0, col0 = position_random(pos2) #pick a bacteria to turn into a hole(kill)
    print("p0:",p0)
    #if condition returns False, AssertionError is raised:
    assert lattice[p0[0],p0[1]] == 2
    
    #print(p0)
    #ID_in= lattice[pb[0], pb[1]] #change it in the lattice

    ID_H = 0
    Efin_H = energy(lattice, ID_H, p0, eps) + muB # energy from interaction and chemical if placed
    #ID_in = lattice[pb[0], pb[1]]
    ID_full = 2
    Ein_H = energy(lattice,ID_full, p0, eps) #evaluate neighbouring energy before
    
    #print("Efin , Ein", Efin, Ein)

    Ediff_H =  Ein_H- Efin_H 
    #print("Ediff ", Ediff)
    
    #Minimum_Energy = max(-Ediff_B, -Ediff_H) #add minus because the energies are negative
    add, remove, move = False, False,False
    if Ediff_H< 0 :
        remove = True

    else:
        probability = np.exp(-Ediff_H/T)
        #print('p_B:', probability)
        if random.random() < probability: # random.random gives between 0 and 1. Hence higher prob -> more move
            remove = True 
        else:
            remove = False
        
    if remove: 
        print(pos2)
        print("remove")
        lattice[p0[0], p0[1]] = 0
        #print("before",pos2, pb)
        # pos0 = np.delete(pos0, colb, axis=1)
         #print('before', pos0, colb)
         
        # Shift the values in columns from col0+1 to the end one position to the left.
        pos2 [:,col0:-1] = pos2[:,col0+1:]
        #print('after', pos0)
        first_negative_column = np.where(np.any(pos0 < 0, axis=0))[0][0]
        # Find the index of the first negative value along the columns in pos0.
        
        pos0[:,first_negative_column] = p0
        print(pos2)
        E_total = E_total + Ediff_H
        #print("Ediff:", Ediff)
        
    else:
        lattice[p0[0], p0[1]] = 2
    #print(E_total, Ediff)
    return lattice, pos2, pos0, E_total


#%%


# check if object moves. pos1 is the coordinates of all objects where one is to be moved. 
# most likely a Tcell
# pos0 are coordinates of holes in the lattice 

def evaluate_particle_moveT(lattice, pos1, pos0, T, E_total, eps):
   
    p1, col1 = position_random(pos1)
    
    ID_in = lattice[p1[0], p1[1]]
    assert ID_in == 1, f"ID_in not 1 but {ID_in}"
    Ein = energy(lattice,ID_in, p1, eps)
    
    p0, col0 = position_random(pos0)
    assert lattice[p0[0],p0[1]] == 0 , f"Found position is not empty"
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
        assert lattice[pos1[0][col1],pos1[1][col1]] == 1, f"not 1 but: {lattice[pos1[0][col1],pos1[1][col1]]}"
        assert lattice[pos0[0][col0],pos0[1][col0]] == 0, f"not 0 but: {lattice[pos0[0][col0],pos0[1][col0]]}"
        E_total = E_total + Ediff
        #print("Ediff:", Ediff)
        
    else:
        lattice[p0[0], p0[1]] = 0
        lattice[p1[0], p1[1]] = 1
    #print(E_total, Ediff)
    return lattice, pos1, pos0, E_total
    
"""
def evaluate_particle_moveT(lattice, pos1, pos0, T, E_total, eps):
    #print("pos1before",pos1.shape)
    p1, col1 = position_random(pos1)
    
    ID_in = lattice[p1[0], p1[1]]
    ID_in = int(ID_in)
    Ein = energy(lattice,ID_in, p1, eps)
    
    p0, col0 = position_random(pos0)
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
        lattice[p0[0], p0[1]] = 1
        # update arrays containing coordinates of 0's and 1's
        pos1[:,col1] = [p0[0], p0[1]]
        pos0[:,col0] = [p1[0], p1[1]]
        #print(pos1)
        E_total = E_total + Ediff
        #print("Ediff:", Ediff)
        
    else:
        lattice[p0[0], p0[1]] = 0
        lattice[p1[0], p1[1]] = 1
    #print(E_total, Ediff)
    return lattice, pos1, pos0, E_total
"""

#%%

def gridprint(lattice):
    lattice_length = len(lattice)
    cmap = plt.cm.colors.ListedColormap(['blue', 'white', 'red'])
    plt.imshow(lattice, cmap=cmap, extent=[0, lattice_length, 0, lattice_length])
    plt.colorbar(ticks=[1, 0, 2], label="Legend")
    plt.title("Lattice with T's (Blue) and B's (Red)")
    #plt.grid(True, linewidth=0.5, color='black')
    plt.show()

def Attempt_moveB(lattice, pos2, pos0, t, E_lattice, eps, muB, B_number):
    add, remove, move = False, False, True
    while move:
        remove = bool(random.getrandbits(1))
        print(remove)
        if remove and B_number>0:
          print("remove", B_number)
          lattice, pos2, pos0, E_lattice = evaluate_particle_removeB(lattice, pos2, pos0, t, E_lattice, eps, muB)
        else:
            print("add")
            lattice, pos2, pos0, E_lattice = evaluate_particle_addB(lattice, pos2, pos0, t, E_lattice, eps, muB)
            add = True
            B_number = B_number +1
    move = not (remove and B_number > 0) and not add
    return lattice, pos2, pos0, E_lattice
###MAKE REMOVE BACTERIA FUNCTION!!!
#%%
def monte_carlo(Temp, eps, lattice_length, T_num_in, B_num_in, muT, muB, num_runs, num_lattices_to_store=None):
 
    E_history = {}
    Tcell = np.zeros(len(Temp))
    B_num = np.zeros(len(Temp))
    #pos2t= []
    for ind, t in enumerate(Temp):
        E_history_for_Temp = []
        lattice, pos1, pos2, pos0 = create_lattice(lattice_length, T_num_in, B_num_in)
        E_lattice = lattice_energy(lattice, eps, muT, muB)
        
        for i in range(0,num_runs): # change to from one and append initial E and lattice to outisde
            E_history_for_Temp.append(E_lattice)
            
            B_number =np.sum((pos2 != -1).all(axis=0))
            #print(B_number)
            if np.all(pos0 < 0):
                #print(pos0)
                print("Lattice is full at iteration:", i)
            else:
                lattice, pos1, pos0, E_lattice = evaluate_particle_moveT(
                                                lattice, pos1, pos0, t, E_lattice, eps)
                lattice, pos2, pos0, E_lattice = Attempt_moveB(lattice, pos2, pos0, t, E_lattice, eps, muB, B_number)
                #lattice, pos2, pos0, E_lattice = evaluate_particle_addB(lattice, pos2, pos0, t, E_lattice, eps, muB) 
                
                #lattice, pos2, pos0, E_lattice = evaluate_particle_removeB(lattice, pos2, pos0, t, E_lattice, eps, muB)  
           
        #pos2t.append(pos2.shape[1])
        #gridprint(lattice)
            #pos0t.append(pos0)
            #pos1t.append(pos1)
        
        #the number of columns without -1 are the number of particles
        Tcell[ind] = np.sum((pos1 != -1).all(axis=0))
        B_num[ind] = np.sum((pos2 != -1).all(axis=0))
        #Tcell.append(pos1.shape[1])   
        print(B_num)
        E_history[t] = E_history_for_Temp.copy()
        
        #pos0_hist.append(pos0t)
        #pos1_hist.append(pos1t)
        #pos2_hist.append(pos2t)
        
    # Unique name for data file 
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime('%Y%m%d-%H-%M')    
    run_name = f'{datetime_str}'
    
    return lattice, E_history, B_num, run_name #, pos0_hist, pos1_hist, pos2_hist

"""
def monte_carlo(Temp, eps, lattice_length, T_num_in, B_num_in, muT, muB, num_runs, num_lattices_to_store=None):
    
    
    
    #print('p0:',pos0,'T:',pos1,'B:',pos2)

    E_history = {}
    #pos0_hist=[]
    #pos1_hist=[]
    #pos2_hist=[]
    Tcell = []
    B_num = np.zeros(len(Temp))
    pos2t= []
    for ind, t in enumerate(Temp):
        #lattice_history = []
        E_history_for_Temp = []
        #pos0t=[]
        #pos1t=[]
        lattice, pos1, pos2, pos0 = create_lattice(lattice_length, T_num_in, B_num_in)
        E_lattice = lattice_energy(lattice, eps, muT, muB)
        
        #gridprint(lattice,lattice_length)
        for i in range(0,num_runs): # change to from one and append initial E and lattice to outisde
            E_history_for_Temp.append(E_lattice)
            
            if any(pos0[1]):
                lattice, pos1, pos0, E_lattice = evaluate_particle_moveT(
                                                lattice, pos1, pos0, t, E_lattice, eps)
            
                lattice, pos2, pos0, E_lattice = evaluate_particle_addB(
                                                lattice, pos2, pos1, pos0, t, E_lattice, eps, muT, muB)
                
        pos2t.append(pos2.shape[1])
                #gridprint(lattice,lattice_length)
            #pos0t.append(pos0)
            #pos1t.append(pos1)
        
            
        B_num[ind] = pos2[1].size
        Tcell.append(pos1.shape[1])   

        E_history[t] = E_history_for_Temp.copy()
        #gridprint(lattice,lattice_length)
        
        #pos0_hist.append(pos0t)
        #pos1_hist.append(pos1t)
        #pos2_hist.append(pos2t)
        
    # Unique name for data file 
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime('%Y%m%d-%H-%M')    
    run_name = f'{datetime_str}'
    
    return lattice, E_history, B_num, pos2t, run_name #, pos0_hist, pos1_hist, pos2_hist
    
"""

#%%


# the interaction matrix can be used to decide how many bacteria are able to multiply. 
# if surrounded by T cells -> no division
# the body is modelled by an N by N lattice

num_runs = 100
#Temp = 0.2
T = np.arange(20,0.01,-4)
#T = np.arange(.1,.01,-0.1) ##Test
size = 10

T_num_in = int(size**2/2)    # number of initial T-cells
B_num_in = 1
muT, muB = -1, -1

BB_int = 1      # interaction energy between bacterias
TT_int = -1      # interaction energy between T-cells
BT_int = 4     # interaction energy between bacteria and T-cells
interaction_matrix = np.array([
    [0, 0, 0],
    [0, TT_int, BT_int],
    [0, BT_int, BB_int]
])

lattice, E_history, B_num, run_name = monte_carlo(T, interaction_matrix, size, T_num_in, B_num_in, muT, muB, num_runs, num_lattices_to_store=None)

#%%


plt.figure()
plt.plot(T,B_num,'.')
plt.xlabel('T')
plt.ylabel('Number of B')
plt.ylim(1240,1260)
plt.show()


#%%

def mean_energy(T, E_history, ind_equilibrium):
    
    E = E_history
    E_mean = np.zeros([len(T)])
    E_variance = np.zeros([len(T)])

    for ind,t in enumerate(T):
        E_mean[ind] = np.mean(E[t][ind_equilibrium:-1])
        E_variance[ind] = np.var(E[t][ind_equilibrium:-1])

    return E_mean, E_variance

#ind_equi = int(0.5*num_runs) 
ind_equi = int((3/8)*num_runs) # index where equilibrium is assumed. 
E_mean, E_var = mean_energy(T, E_history, ind_equi)

#%%
def E_history_plot(E_history, T, num_runs):

    yMin = min(E_history)-1
    yMax = max(E_history)

    E_keys = list(E_history.keys())
    # Create subplots
    fig, axes = plt.subplots(len(T), 1, figsize=(8, 2*len(T)))
    fig.suptitle(f'hi there')
    for i in range(len(T)):
        ax = axes[i]
        
        T_formatted = f'{T[i]:.2f}'
        ax.plot(np.arange(0, num_runs), E_history[E_keys[i]], '.', markersize = '1')
        ax.set_yticks(np.arange(yMin, yMax, step = 20))
        ax.set_ylabel(f'Energy (T = {T_formatted})')
        #ax.set_xlabel('Iteration')
        #ax.set_title()

    plt.show()


#E_history_plot(E_history, T, num_runs)
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

file_spec = '1e4_test'
file_name = f'{run_name}_{file_spec}.npz'

np.savez(file_name, 
         T = T,
         eps = interaction_matrix,
         muT = muT,
         muB = muB,
         Tcell_num = T_num_in,
         B_num = B_num,
         size = size,
         E_mean = E_mean,
         E_variance = E_var,
         num_runs = num_runs,        
        )

