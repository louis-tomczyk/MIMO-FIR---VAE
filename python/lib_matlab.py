# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.1.0
#   Date            : 2024-05-24
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2021-09-29) - creation
#   1.0.1 (2023-08-17) - updates
#   1.0.2 (2024-04-24) - enhancements
#   1.1.0 (2024-05-24) - [NEW] numel
#
# ----- MAIN IDEA -----
#   Matlab fashionned functions
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   [A1] Authors        : 
#       Title           : 
#       Journal/Editor  : 
#       Volume - N°     : 
#       Date            : 
#       DOI/ISBN        : 
#       Pages           : 
#  ----------------------
#   CODE
#   [C1] Author         : 
#       Contact         : 
#       Affiliation     : 
#       Date            : 
#       Program Title   : 
#       Code Version    : 
#       Web Address     : 
# ---------------------------------------------
# %%


#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================
import os
import pwd
import stat
import sys
import inspect
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np



#%% ===========================================================================
# --- CONTENTS
# =============================================================================
# - cd
# - clc
# - clear_all
# - imagesc
# - inputname
# - linspace
# - ls_l
# - numel           (1.1.0)
# - PWD
# - repmat
# =============================================================================


#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================


#%%
def cd(path):
    return os.chdir(path)

#%%

# CHATGPT
def clc():
    os.system('cls' if os.name == 'nt' else 'clear')

#%%

# CHATGPT
def clear_all():
    # Obtenez le dictionnaire des variables locales du module principal
    current_module = sys.modules['__main__']
    main_vars = vars(current_module)

    # Supprimez uniquement les variables créées dans le script
    for var_name in list(main_vars.keys()):
        if not var_name.startswith('__') and var_name != 'clear':
            del main_vars[var_name]


#%%

# CHATGPT
def imagesc(matrix, cmap='viridis', interpolation='nearest', origin='upper', aspect='auto'):
    plt.imshow(matrix, cmap=cmap, interpolation=interpolation, origin=origin, aspect=aspect)
    plt.colorbar()
    plt.show()


#%%

def inputname(var):
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame)[1]
    caller_locals = caller_frame.frame.f_locals
    
    for name, value in caller_locals.items():
        if id(value) == id(var):
            return name
    
    return None


#%%
def linspace(start,end,numel,axis = "col"):

    array = np.linspace(start,end,numel)
    array = array[:,np.newaxis]
    if axis != "col":
        array = array.transpose()
    
    return array

#%%

# CHATGPT
def ls_l(path="."):
    files = os.listdir(path)

    for file in files:
        file_path           = os.path.join(path, file)
        stat_info           = os.stat(file_path)
        permissions         = stat.filemode(stat_info.st_mode)
        owner               = pwd.getpwuid(stat_info.st_uid).pw_name
        size                = stat_info.st_size
        modification_time   = datetime.datetime.fromtimestamp(stat_info.st_mtime)
        
        print(f"{permissions} {owner} {size} {modification_time} {file}")



#%%
def numel(x):
    numel   = 1
    shape   = np.shape(x)

    for k in range(len(shape)):
        numel = numel*shape[k]

    return numel


#%%

def PWD(show = True):
    if show == True:
        print(os.getcwd())
    return os.getcwd()


#%%
def repmat(input,shape):
    
    return np.tile(input,[int(shape[k]) for k in range(len(shape))])
