import library_louis as lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

            # ================================================ #
            # ================================================ #
            # ================================================ #


def find_start_test_index(Thetas):
    
    Thetas_roll = np.roll(Thetas,-1)
    Delta_Thetas = Thetas_roll-Thetas
    index = len(np.where(Delta_Thetas == 0)[0])+1
    
    return index
    
def get_thetas_shift(Thetas_test):
    
    Thetas_test_roll = np.roll(Thetas_test,-1)
    Delta_Thetas = Thetas_test_roll-Thetas_test
    
    return Delta_Thetas[:-1]
    
    
            # ================================================ #
            # ================================================ #
            # ================================================ #
            
            

data,field_names,filenames,filepaths  = lib.select_and_read_files()
Ndata       = len(data[0][0])

Iterations  = data[0][:,0]
Losses      = data[0][:,1]
SNRs        = data[0][:,2]
SERs        = data[0][:,3]
Thetas      = data[0][:,4] # degrees
Data        = np.concatenate((Iterations, Losses, SNRs,Thetas),axis = 0)

index_test  = find_start_test_index(Thetas)
Thetas_test = Thetas[index_test:]
Delta_Thetas= get_thetas_shift(Thetas_test)

plt.figure
nbins = 20
plt.hist(Delta_Thetas,nbins,density= True)













'''

len_filename = len(filenames[0])
path_to_save = filepaths[0][:-len_filename-4]

PWD = lib.PWD()
lib.cd(path_to_save)
df.to_csv("sum up {}.csv".format(field_names[0]),index = False)
lib.cd(PWD)
'''

    
    
    