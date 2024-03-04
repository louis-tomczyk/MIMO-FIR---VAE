# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          :
#   Date            : 2023-03-04
#   Version         : 1.0.0
#   Licence         : cc-by-nc-sa
#                     Attribution - Non-Commercial - Share Alike 4.0 International
# 
# ----- Main idea -----
# ----- INPUTS -----
# ----- BIBLIOGRAPHY -----
#   Articles/Books
#   Authors             : 
#   Title               :
#   Jounal/Editor       : 
#   Volume - N°         : 
#   Date                :
#   DOI/ISBN            :
#   Pages               :
#  ----------------------
#   Functions           : 
#   Author              : 
#   Author contact      : 
#   Affiliation         :
#   Date                : 
#   Title of program    : 
#   Code version        : 
#   Type                :
#   Web Address         :
# ---------------------------------------------
# %%


import os
import pwd
import stat
import sys
from datetime import datetime




            # ================================================ #
            # ================================================ #
            # ================================================ #

def PWD(show = True):
    if show == True:
        print(os.getcwd())
    return os.getcwd()
    
            # ================================================ #
            # ================================================ #
            # ================================================ #
            
def cd(path):
    return os.chdir(path)

            # ================================================ #
            # ================================================ #
            # ================================================ #

# CHATGPT
def clc():
    os.system('cls' if os.name == 'nt' else 'clear')

            # ================================================ #
            # ================================================ #
            # ================================================ #

# CHATGPT
def clear_all():
    # Obtenez le dictionnaire des variables locales du module principal
    current_module = sys.modules['__main__']
    main_vars = vars(current_module)

    # Supprimez uniquement les variables créées dans le script
    for var_name in list(main_vars.keys()):
        if not var_name.startswith('__') and var_name != 'clear':
            del main_vars[var_name]



            # ================================================ #
            # ================================================ #
            # ================================================ #

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

            # ================================================ #
            # ================================================ #
            # ================================================ #