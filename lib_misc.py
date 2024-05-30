# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2023-03-04 (1.0.0) creation
#                   : 2024-04-01 (1.1.1) [NEW] string_to_binary / binary_to_decimal
#                   : 2024-04-03 (1.1.2) [NEW] plot_1y_axes
#                   : 2024-05-22 (1.2.0) [NEW] convert_byte, get_total_size
#   Date            : 2024-05-28 (1.2.1) create_xml_file 
#   Version         : 1.2.1
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute - place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source - include copyright - state changes - include license
# 
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
#   Functions 
#   Author              : [C1] CNRS - IDRIS
#   Author contact      :
#   Affiliation         :
#   Date                : 2023-05-09
#   Title of program    : Profilage de codes python
#   Code version        :
#   Type                : tutorial
#   Web Address         : http://www.idris.fr/jean-zay/cpu/jean-zay-cpu-python.html
# ---------------------------------------------
# %%


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import tkinter as tk
import torch
from datetime import date
from datetime import datetime
import scipy.io as io
import csv
import sys
from collections import deque
from itertools import chain

import xml.etree.ElementTree as ET
from dateutil.parser import parse
from tkinter import filedialog


import lib_general as gen
import lib_matlab as mat
import lib_misc as misc
pi = np.pi


#%% ============================================================================
# --- CONTENTS ---
# - are_same_dicts
# - are_same_tensors
# - array2csv
# - binary_to_decimal               (1.1.1)
# - create_xml_file                 (1.3.0)
# - convert_byte                    (1.2.0)
# - extract_values_from_filename
# - find_string
# - get_total_size                  (1.2.0)
# - import_data
# - init_dict
# - is_date
# - KEYS
# - list2vector
# - list_folders
# - list_functions
# - merge_data_folders
# - move_files_to_folder
# - my_tensor
# - my_zeros_tensor
# - plot_1y_axes                    (1.1.2)
# - plot_2y_axes
# - plot_3y_axes
# - remove_using_index
# - save2mat
# - select_and_read_files
# - set_saving_params
# - show_dict
# - sort_dict_by_keys
# - sort_key
# - string_to_binary                (1.1.1)
# - what
# - xml2dict
# =============================================================================

#%%

fig_width                       = 10
fig_height                      = fig_width/1.618
fig_resolution                  = fig_width*fig_height


plt.rcParams['figure.figsize']  = (fig_width, fig_height*2)
plt.rcParams['figure.dpi']      = fig_resolution*5
plt.rcParams['font.weight']     = "normal"
plt.rcParams['axes.labelweight']= "bold"
plt.rcParams['axes.linewidth']  = "0.1"
plt.rcParams["axes.titlesize"]  = "20"
plt.rcParams["axes.labelsize"]  = "12"
plt.rcParams["axes.titleweight"]= "bold"



#%% ============================================================================
# --- FUNCTIONS ---
#% =============================================================================


#%%

def are_same_dicts(dict1,dict2):
    
    keys_unequal = []
    
    for key in dict1:
        flag = np.array_equal(dict1[key], dict2[key])
        if flag == False:
            keys_unequal.append(key)

    return keys_unequal


#%%
def are_same_tensors(tenseur1, tenseur2):

    if tenseur1.shape == tenseur2.shape:

        sameOnes = torch.all(tenseur1 == tenseur2).item() == 1
        return sameOnes
    
    else:
        return False


#%%

# GPT
def array2csv(input, filename, *varargin):
    
    if len(varargin) != 0:

        dataFrame = pd.DataFrame(input, columns=varargin[0])
        
    else:
        dataFrame = pd.DataFrame(input)
        
    dataFrame.fillna(0, inplace=True)
    dataFrame.replace([np.inf, -np.inf], 0, inplace=True)
    dataFrame.to_csv(filename + '.csv', index=False)



#%%
# ChatGPT
def binary_to_decimal(binary_string):
    decimal_number = 0
    power = len(binary_string) - 1
    for bit in binary_string:
        if bit == '1':
            decimal_number += 2 ** power
        power -= 1
    return decimal_number


#%% [C1]
def convert_byte(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

#%%

# ChatGPT
def create_xml_file(tx,fibre,rx,saving):
    
    sections    = ["TX","CHANNEL","RX"]
    
    if tx['nu'] != 0:
        TX      = ["mod","nu","Nsps", "Rs","Ntaps",'dnu','SNRdB']
    else:
        TX      = ["mod", "Nsps", "Rs","Ntaps",'dnu','SNRdB']

    CHANNEL     = ["tauPMD", "tauCD", "kind","law"]
    RX          = ["mimo",'lr',"Nframes", "NSymbBatch", "FrameChannel", "NSymbFrame","SNR_dB"]
    fields_list = [TX, CHANNEL, RX]
    
    if tx['nu'] != 0:
        saving_list = ["mimo",'lr','Rs','mod',"nu",'dnu','SNRdB',"CD","PMD",'kind','law',"NSymbFrame","NSymbBatch","SNR_dB","Ntaps"]
    else:
        saving_list = ["mimo",'lr','Rs','mod','dnu','SNRdB',"CD","PMD",'kind','law',"NSymbFrame","NSymbBatch","SNR_dB","Ntaps"]
        
    CHANNELpar  = [np.round(fibre["tauPMD"]*1e12,0),           # [ps]
                   np.round(np.sqrt(fibre["tauCD"])*1e12,0),   # [ps]
                   fibre["ThetasLaw"]["kind"],
                   fibre["ThetasLaw"]["law"]]
    
    if fibre["ThetasLaw"]["kind"] == "Rwalk":
        
        if fibre["ThetasLaw"]["law"] == "uni":
            saving_list.insert(6,'low')
            saving_list.insert(7,'high')
            
            CHANNEL.append('low')
            CHANNEL.append('high')
            
            CHANNELpar.append(fibre["ThetasLaw"]['low']*180/pi)
            CHANNELpar.append(fibre["ThetasLaw"]['high']*180/pi)
        
        if fibre["ThetasLaw"]["law"] == "gauss":
            saving_list.insert(6,'theta_in')
            saving_list.insert(7,'theta_std')
            
            CHANNEL.append('theta_in')
            CHANNEL.append('theta_std')
            
            CHANNELpar.append(np.round(fibre["ThetasLaw"]['theta_in']*180/np.pi,0))
            CHANNELpar.append(np.round(fibre["ThetasLaw"]['theta_std']*180/np.pi,0))
            
        
        if fibre["ThetasLaw"]["law"] == "tri":
            saving_list.insert(6,'low')
            saving_list.insert(7,'mode')
            saving_list.insert(8,'high')
            
            CHANNEL.append('low')
            CHANNEL.append('mode')
            CHANNEL.append('high')
            
            CHANNELpar.append(np.round(fibre["ThetasLaw"]['low']*180/np.pi,0))
            CHANNELpar.append(np.round(fibre["ThetasLaw"]['mode']*180/np.pi,0))
            CHANNELpar.append(np.round(fibre["ThetasLaw"]['high']*180/np.pi,0))
            
    elif fibre["ThetasLaw"]["kind"] == "func":
        if fibre["ThetasLaw"]["law"] == "lin":
            saving_list.insert(11,'End')
            saving_list.insert(12,'Slope')

            CHANNEL.append('End')
            CHANNEL.append('Slope')
            
            Slope = (fibre["ThetasLaw"]['End']-fibre["ThetasLaw"]['Start'])/(rx['Nframes']-rx['FrameChannel']) # [rad]
            CHANNELpar.append(np.round(fibre["ThetasLaw"]['End']*180/pi,1))
            CHANNELpar.append(np.round(Slope*180/pi,2))
            
            print(Slope*180/pi)



    tx["Ntaps"] = 2*tx['NSymbTaps']+1


    TXpar       = [tx["mod"],
                   tx["Nsps"],
                   int(tx["Rs"]*1e-9),
                   tx["Ntaps"],
                   int(tx["dnu"]*1e-3),
                   tx['SNRdB']]
    
    
    if tx["nu"] != 0:
        TXpar.insert(1, tx['nu'])
    

    
    RXpar       = [rx['mimo'],
                   rx['lr']*1000,
                   rx["Nframes"],
                   rx["NSymbBatch"],
                   rx["FrameChannel"],
                   np.round(rx["NSymbFrame"]*1e-3,2),
                   rx['SNRdB']]
    
    params_list             = [TXpar, CHANNELpar,RXpar]
    [base_path, my_path]    = set_saving_params()
    
    os.chdir(my_path)
    
    # root element for XML
    root = ET.Element("SIMULATION-PARAMETERS")

    # create the sub sections and their fields from the given lists
    for section, section_params in zip(sections, params_list):
        section_element = ET.SubElement(root, section)
        
        for field_name, field_value in zip(fields_list[sections.index(section)], section_params):
            # use the field name as the element name
            field_element       = ET.SubElement(section_element, field_name)
            field_element.text  = str(field_value)

    # file name creation
    current_time    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name       = current_time

    for name in saving_list:
        # print("\n"*3)
        # print(name)
 
        # find the index of the field in fields_list
        for section_fields in fields_list:
            # print(section_fields)
            
            if name in section_fields:
                # print(name)
                index = section_fields.index(name)
            
                # Uses the index to get the value corresponding to the field in params_lists
                value = params_list[fields_list.index(section_fields)][index]
                # print(value)
                file_name += f" - {name} {value}"

    file_name += ".xml"

    # create the xml file
    tree = ET.ElementTree(root)
    tree.write(file_name, encoding="utf-8", xml_declaration=True)
    
    os.chdir("../")

    # print(file_name[:-4])
    return file_name



#%%

# ChatGPT
def extract_values_from_filename(filename):
    values      = {}
    keywords    = []
    parts       = filename.split(' - ')
    
    for part in parts[1:]:
        
        keyword, value_str = part.split()
        values[keyword]    = value_str
        
        keywords.append(keyword)
    
    value_with_ext = values[keywords[-1]]
    
    if "." in value_with_ext:
        
        value_without_ext       = value_with_ext[:-4]
        values[keywords[-1]]    = value_without_ext
    
    return values,keywords


#%%

def find_string(pattern, mystring):
    
    pattern_ind     = [0,0]
    pattern_ind[0]  = mystring.find(pattern)
    Nchars          = len(pattern)
    pattern_ind[1]  = pattern_ind[0]+Nchars
    
    return pattern_ind



#%% [C1] + ChatGPT

def get_total_size(object_name, o, handlers={}, verbose=False, verbose_dict=False):
    """ Returns the approximate memory footprint of an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses: tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    Put FALSE to both verbose and verbose_dict to only display the total memory and not
    the sub-elements
    """
    
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o, obj_name="object"):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(f"________ Memory consumption of {obj_name} ({type(o)}) {convert_byte(s)}")

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                if isinstance(o, dict) and verbose_dict:
                    for k, v in o.items():
                        s += sizeof(k, f"{obj_name}.key({repr(k)})")
                        s += sizeof(v, f"{obj_name}[{repr(k)}]")
                else:
                    s += sum(map(sizeof, handler(o)))
                break
        return s

    total_memory = sizeof(o, object_name)
    print(f"Total memory consumption of {object_name}: {convert_byte(total_memory)}")
    
    return total_memory
#%%
            
def import_data(Nsps = 2, scale = 1):
    
    data            = dict()
    data["tx_real"] = select_and_read_files(extension = "csv",skiprow=0,\
                                            title="select TX data in </data_from_matlab/*X*.csv")[0][0]
    data["rx_real"] = select_and_read_files(extension = "csv",skiprow=0,\
                                            title="select RX data in </data_from_matlab/*Y*.csv")[0][0]

    XHI             = data["tx_real"][0]
    XHQ             = data["tx_real"][1]
    XVI             = data["tx_real"][2]
    XVQ             = data["tx_real"][3]
    
    YHI             = data["rx_real"][0]
    YHQ             = data["rx_real"][1]
    YVI             = data["rx_real"][2]
    YVQ             = data["rx_real"][3]

    data["tx_real"] = my_tensor(np.array([[XHI,XHQ],[XVI,XVQ]]))
    data["rx_real"] = my_tensor(np.array([[YHI,YHQ],[YVI,YVQ]]))
            
    gen.plot_const_2pol(data['tx_real'],"tx")
    gen.plot_const_2pol(data['rx_real'],"rx")

    return data
        


#%%            
def init_dict():
    
    tx          = dict()
    fibre       = dict()
    rx          = dict()
    saving      = dict()
    flags       = dict()

    fibre['ThetasLaw']  = dict()
    tx["PhiLaw"]        = dict()

    for field_name in ["mod","Nsps","Rs","nu","Ntaps",'dnu']:
        tx[field_name] = 0
        
    #for field_name in ["channel",'PMD','CD','phiIQ','theta1','theta_std']:
    for field_name in ['PMD','CD','phiIQ']:
        fibre[field_name] = 0
    
        
    for field_name in ['SNRdB',"NSymbBatch","N_lrhalf","Nframes","FrameChannel","Nmax_SymbFrame","mimo"]:
        rx[field_name] = 0
        
    rx["Frame"]             = 0
    tx["RollOff"]           = 0.1                   # roll-off factor
    tx["Nsps"]              = 2                     # oversampling factor (Shannon Coefficient) in samples per symbol
    tx["nu"]                = 0                     # [0] [0.0270955] [0.0872449] [0.1222578]
    rx["N_lrhalf"]          = 170

    saving['root_path'] = mat.PWD(show = False)
    saving['merge_path']= saving['root_path']+'/data-'+str(date.today())

    tx      = sort_dict_by_keys(tx)
    fibre   = sort_dict_by_keys(fibre)
    rx      = sort_dict_by_keys(rx)
    saving  = sort_dict_by_keys(saving)
    
    flags["plot"] = False
    
    return tx,fibre,rx,saving,flags

#%%
                
# https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
def is_date(string,fuzzy=False):
    """
    Return whether the string can be interpreted as a date.
    
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True
    
    except ValueError:
        return False
                

#%%

def KEYS(dict):
    for key in dict.keys():
        print(key)

    
#%%
            
def list2vector(input,axis = 'col'):

    array = np.array(input)
    array = array[:,np.newaxis]
    if axis != "col":
        array = array.transpose()
        
    return array



#%%
def list_folders(path):
    try:
        # Utilisez la fonction os.listdir() pour obtenir la liste de tous les fichiers et répertoires dans le path spécifié.
        folders = os.listdir(path)

        # Filtrez les éléments qui sont des répertoires en utilisant os.path.isdir().
        folders = [folder for folder in folders if os.path.isdir(os.path.join(path, folder))]

        return folders
    
    except OSError as e:
        # Gérez les erreurs liées au path spécifié (par exemple, s'il n'existe pas).
        print(f"Erreur : {e}")
        return []
        
#%%
def list_functions(module_name):
    try:
        # Import the specified module by name
        module = __import__(module_name)
        print(f"The functions in module {module_name} are:")
        # Get all attributes of the module
        attributes = dir(module)
        # Sort attributes alphabetically ignoring case
        sorted_attributes = sorted(attributes, key=lambda x: x.lower())
        # Iterate through sorted attributes and display those that are functions
        for attribute_name in sorted_attributes:
            attribute = getattr(module, attribute_name)
            # Check if the attribute is a function
            if callable(attribute):
                print(f"- {attribute_name}")
    except ImportError:
        print(f"The module {module_name} could not be imported.")
    
            
            
#%%

def merge_data_folders(saving, deletion=True):
    # Liste tous les fichiers dans le répertoire courant
    folders         = list_folders(saving['root_path'])
            
    folders_tmp = []
    for folder in folders:
        if folder[0:5] == 'data-':
            folders_tmp.append(folder)
   
    folders             = sorted(folders_tmp,key=sort_key)
    Nfolders            = len(folders)
    merge_folder        = saving["merge_path"][-(4+4+2+2+3):] # 4 for 'data' and '2023', 2 for 'yy', 'mm', 'dd', 3*1 for '-'
    folders_to_delete   = []
    
    for k in range(Nfolders):
        if (merge_folder in folders[k]) & (merge_folder != folders[k]):
            folders_to_delete.append(folders[k])

    folders_to_delete = sorted(folders_to_delete,key=sort_key)

    for folder in folders_to_delete:
        source_folder = os.path.join(saving['root_path'], folder)
        
        for file_name in os.listdir(source_folder):
            # Vérifier si le fichier a la même date que le dossier de fusion
            merge_folder_tmp = merge_folder[4+1+2:] # 4 for 'data', 1 for '-', 2 for '20'
            
            if file_name.startswith(merge_folder_tmp):
                source      = os.path.join(source_folder, file_name)
                destination = os.path.join(saving['merge_path'], file_name)
                
                if os.path.isfile(source):
                    shutil.move(source, destination)
                    print('moved', file_name)

    if deletion:
        for folder in folders_to_delete:
            path_tmp = os.path.join(saving['root_path'], folder)
            shutil.rmtree(path_tmp)


#%%


def move_files_to_folder():
    # Obtenez la liste des fichiers dans le répertoire actuel
    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    for filename in files:

        # Vérifiez si le nom de fichier suit le format date attendu
        parts       = filename.split(' - ')
        date_part   = parts[0].split(' ')
        day_part    = date_part[0]

        if is_date(day_part) == True:
            if int(day_part[0]) == 20:
                date_folder = f'data-{day_part}'
            else:
                date_folder = f'data-20{day_part}'
                            
            # Vérifiez si le dossier existe déjà, sinon, créez-le
            if not os.path.exists(date_folder):
                os.mkdir(date_folder)
            # Déplacez le fichier dans le dossier approprié
            shutil.move(filename, os.path.join(date_folder, filename))
                
      
#%%

def my_tensor(size,device='cpu',dtype=torch.float32,requires_grad=False):
    return torch.tensor(size,device=device,dtype=dtype,requires_grad=requires_grad)



#%%

def my_zeros_tensor(size,device='cpu',dtype=torch.float32,requires_grad=False):
    return torch.zeros(size,device=device,dtype=dtype,requires_grad=requires_grad)


#%%

def plot_1y_axes(saving,xaxis,yaxis,extensions,*varargin):
    

    directory_path = saving["root_path"]
    # Vérifier si le répertoire existe
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The folder '{directory_path}' does not exists.")

    # Liste des fichiers CSV dans le répertoire
    csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

    # Parcourir tous les fichiers CSV dans le répertoire
    for csv_file in csv_files:
        csv_path = os.path.join(directory_path, csv_file)

        # Extraire les valeurs des mots-clés du nom du fichier
        values, keywords = extract_values_from_filename(csv_file)

        # Charger le fichier CSV en utilisant pandas
        df = pd.read_csv(csv_path)
        x  = df[xaxis]
        y  = df[yaxis]
        
        if len(varargin) == 1:
            x       = x[varargin[0]:]
            y       = y[varargin[0]:]

        # Créer la figure
        fig, ax1 = plt.subplots(figsize=(10, 6.1423))

        # Tracer y1 en fonction d'itération sur l'axe gauche
        ax1.plot(x, y, color='tab:blue',linestyle='dashed',linewidth = 2)
        ax1.set_xlabel(xaxis)
        ax1.set_ylabel(yaxis, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        
        mytext  = ''.join(["{}:{} - ".format(key,values[key]) for key in keywords if key in values])
        
        text_lim = 45
        
        if len(mytext)>text_lim:

            mytext2 = mytext[text_lim:]
            mytext  = mytext[:text_lim]
 
            if len(mytext2)>text_lim:
    
                mytext3 = mytext2[text_lim+11:]
                mytext2 = mytext2[:text_lim+11]
            
            pos_mytext  = 0.25#len(mytext)/50
            pos_mytext2 = 0.25#len(mytext2)/50            
            pos_mytext3 = 0.25#len(mytext3)/50
            
        plt.text(0.5-pos_mytext, 1, mytext, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.5-pos_mytext2, 0.95, mytext2, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.5-pos_mytext3, 0.9, mytext3, fontsize=14, transform=plt.gcf().transFigure)
            
        # tmp     = len(mytext)/200
        # plt.text(0.5-tmp, 0.95, mytext, fontsize=14, transform=plt.gcf().transFigure)
        
        # Sauvegarder la figure en format image
        if type(extensions) == list:
            for extension in extensions:
                output_file = os.path.splitext(csv_file)[0] + '.'+ extension
                plt.savefig(output_file, bbox_inches='tight')
        else:
            output_file = os.path.splitext(csv_file)[0] + '.'+ extensions
            plt.savefig(output_file, bbox_inches='tight')
            

#%%


def plot_2y_axes(saving,xaxis,yaxis_left,yaxis_right,extensions,*varargin):
    

    directory_path = saving["root_path"]
    # Vérifier si le répertoire existe
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The folder '{directory_path}' does not exists.")

    # Liste des fichiers CSV dans le répertoire
    csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

    # Parcourir tous les fichiers CSV dans le répertoire
    for csv_file in csv_files:
        csv_path = os.path.join(directory_path, csv_file)

        # Extraire les valeurs des mots-clés du nom du fichier
        values, keywords = extract_values_from_filename(csv_file)

        # Charger le fichier CSV en utilisant pandas
        df = pd.read_csv(csv_path)
        x  = df[xaxis]
        y1        = df[yaxis_left]
        y2         = df[yaxis_right]
        
        if len(varargin) == 1:
            x  = x[varargin[0]:]
            y1        = y1[varargin[0]:]
            y2         = y2[varargin[0]:]

        # Créer la figure
        fig, ax1 = plt.subplots(figsize=(10, 6.1423))

        # Tracer y1 en fonction d'itération sur l'axe gauche
        ax1.plot(x, y1, color='tab:blue',linestyle='dashed',linewidth = 2)
        ax1.set_xlabel(xaxis)
        ax1.set_ylabel(yaxis_left   , color='tab:blue')
        # ax1.set_ylim(-1200, 700)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Créer un axe Y droit pour le SNR
        ax2 = ax1.twinx()

        # Tracer SNR en fonction d'itération sur l'axe droit
        ax2.plot(x, y2, color='tab:red')
        # ax2.set_ylabel(yaxis_right, color='tab:red')
        ax2.set_ylabel(yaxis_right, color='tab:red')
        # ax2.set_ylim(0, 40)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        mytext  = ''.join(["{}:{} - ".format(key,values[key]) for key in keywords if key in values])
        
        text_lim = 45
        
        if len(mytext)>text_lim:

            mytext2 = mytext[text_lim:]
            mytext  = mytext[:text_lim]
 
            if len(mytext2)>text_lim:
    
                mytext3 = mytext2[text_lim+11:]
                mytext2 = mytext2[:text_lim+11]
            
            pos_mytext  = 0.25#len(mytext)/50
            pos_mytext2 = 0.25#len(mytext2)/50            
            pos_mytext3 = 0.25#len(mytext3)/50
            
        plt.text(0.5-pos_mytext, 1, mytext, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.5-pos_mytext2, 0.95, mytext2, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.5-pos_mytext3, 0.9, mytext3, fontsize=14, transform=plt.gcf().transFigure)
            
        # tmp     = len(mytext)/200
        # plt.text(0.5-tmp, 0.95, mytext, fontsize=14, transform=plt.gcf().transFigure)
        
        # Sauvegarder la figure en format image
        if type(extensions) == list:
            for extension in extensions:
                output_file = os.path.splitext(csv_file)[0] + '.'+ extension
                plt.savefig(output_file, bbox_inches='tight')
        else:
            output_file = os.path.splitext(csv_file)[0] + '.'+ extensions
            plt.savefig(output_file, bbox_inches='tight')
            
        
#%%

def plot_3y_axes(saving,xaxis,yaxis_left,yaxis_right,yaxis_right_2,extensions):
    
    directory_path = saving["root_path"]
    

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"the folder '{directory_path}' does not exists.")

    csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        csv_path    = os.path.join(directory_path, csv_file)
        df          = pd.read_csv(csv_path)

        values,keys = extract_values_from_filename(csv_file)

        x           = df[xaxis]
        y1          = df[yaxis_left]
        y2          = df[yaxis_right]
        y3          = df[yaxis_right_2]

        fig, ax1    = plt.subplots(figsize=(10, 6.1423))

        ax1.plot(x, y1, color='tab:blue',linestyle='dashed',linewidth = 2)
        ax1.set_xlabel(xaxis)
        ax1.set_ylabel(yaxis_left, color='tab:blue')
#        ax1.set_ylim(-1200, 700)
        locs, labels = plt.xticks()  # Get the current locations and labels.
        plt.xticks(np.arange(0, len(y3)+1, step=5))  # Set label locations.
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.plot(x, y2, color='tab:red')
        ax2.set_ylabel(yaxis_right, color='tab:red')
#        ax2.set_ylim(0, 40)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Ajustement de la position de l'axe
        ax3.set_ylabel(yaxis_right_2, color='tab:green')
        ax3.plot(x, y3, color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
#        ax3.set_ylim(-45,+45)
        
        mytext      = ''.join(["{}:{} - ".format(key,values[key]) for key in keys if key in values])
        text_lim    = 45
        
        if len(mytext)>text_lim:

            mytext2 = mytext[text_lim:]
            mytext  = mytext[:text_lim]
 
            if len(mytext2)>text_lim:
    
                mytext3 = mytext2[text_lim+11:]
                mytext2 = mytext2[:text_lim+11]
            
            pos_mytext  = 0.25#len(mytext)/50
            pos_mytext2 = 0.25#len(mytext2)/50            
            pos_mytext3 = 0.25#len(mytext3)/50
            
        plt.text(0.5-pos_mytext, 1, mytext, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.5-pos_mytext2, 0.95, mytext2, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.5-pos_mytext3, 0.9, mytext3, fontsize=14, transform=plt.gcf().transFigure)
        
        # Sauvegarder la figure en format image
        if type(extensions) == list:
            for extension in extensions:
                output_file = os.path.splitext(csv_file)[0] + '.'+ extension
                plt.savefig(output_file, bbox_inches='tight')
        else:
            output_file = os.path.splitext(csv_file)[0] + '.'+ extensions
            plt.savefig(output_file, bbox_inches='tight')
            
        # plt.close('all')



#%%
        
#   https://www.kite.com/python/answers/how-to-remove-an-element-from-an-array-in-python
def remove_using_index(All_Indexes,List_or_Array):
    
    if type(List_or_Array) == list:
        Type = 'Liste'
    else:
        Type = 'Array'
    List_or_Array = list(List_or_Array)
    
    if type(List_or_Array) == list:
        N_Steps = len(list(All_Indexes))
        for k in range(N_Steps):
            tmp_index = All_Indexes[k]-k
            del(List_or_Array[tmp_index])
    if Type != "Liste":
        List_or_Array = np.array(List_or_Array)
        
    return List_or_Array
    
    

#%%
def save2mat(tx,fibre,rx,saving):


    name        = saving["filename"]+".mat"
    save_dict   = {
                'NtapsTX'           : tx["Ntaps"],
                'nu'                : tx["nu"],
                'nu_sc'             : tx["nu_sc"],
                'Rs'                : tx["Rs"],
#                'theta_in'          : fibre["ThetasLaw"]["theta_in"],
#                'theta_std'         : fibre["ThetasLaw"]["theta_std"],
#                'thetas'            : fibre["thetas"],
                'NSymbBatch'          : rx["NSymbBatch"],
                'h_channel_liste'   : rx["H_lins"],
                'h_est_liste'       : rx["H_est_l"],
                'lr'                : rx["lr"],
                'Losses'            : rx["Losses"],
                'Nframes'           : rx["Nframes"],
                'FrameChannel'      : rx["FrameChannel"],
                'SNRdBs'            : rx["SNRdBs"],
                'SER_means'         : rx["SERs"],
                'SER_valid'         : rx["SER_valid"],
                'Var_est'           : rx["Pnoise_est"],
                }
    
    if rx['mimo'].lower() == "vae":
        save_dict['h_est']      = rx["h_est"].detach().numpy()
        save_dict['Var_real']   = rx["noise_var"].detach().numpy()

    else:
        save_dict['h_est']      = rx["h_est"]
        save_dict['Var_real']   = rx["noise_var"]
        
    io.savemat(name,save_dict)



    
#%%

def select_and_read_files(extension="csv",skiprow = 0,title = "Select files",Multiple = True):
    """
    return[0] = data_matrices
    return[1] = field_names
    return[2] = file_names
    return[3] = file_paths
    """
    
    root = tk.Tk()
    root.withdraw()  # Masque la fenêtre principale de tkinter

    file_paths = filedialog.askopenfilenames(
        title       = title,
        filetypes   = [(f"Files (*.{extension})", f"*.{extension}")],
        multiple    = Multiple
    )

    data_matrices   = []
    file_names      = []
    
    for file_path in file_paths:
        with open(file_path) as csv_file:
            
            csv_reader = csv.reader(csv_file,delimiter=',')
            
            if skiprow > 0:
                for row in csv_reader:
                    field_names = row
                    break
            
        # get the names of the files
        file_name = file_path.split("/")[-1]
        file_names.append(file_name)

        # read files using the delimiter
        if extension == "csv":
            data = np.loadtxt(file_path, delimiter=",", skiprows=skiprow)  # Ignorer la première ligne et utiliser la virgule comme délimiteur
            data_matrices.append(data)

    if skiprow == 0:
        return data_matrices,file_names, file_paths
    else:
        return data_matrices, field_names, file_names, file_paths
    
    


#%%
            
def set_saving_params():
    
    current_day = date.today().strftime("%Y-%m-%d")

    base_path   = "data-"+current_day
    k           = 0
    my_path_tmp = base_path

    while os.path.exists(my_path_tmp):
        k += 1
        my_path_tmp = f"{base_path}-{k}"

    try:
        os.mkdir(my_path_tmp)
        # print(f"Dossier créé : {my_path_tmp}")
    except Exception as e:
        print(f"Erreur lors de la création du dossier : {str(e)}")

    return [base_path, my_path_tmp]




#%%

# ChatGPT
def show_dict(my_dict):
    max_key_length = max(len(key) for key in my_dict.keys())
    
    for key, value in my_dict.items():
        if isinstance(value, list):
            value_str = "[" + ", ".join(map(str, value)) + "]"
        else:
            value_str = str(value)
        
        formatted_key = key.ljust(max_key_length)
        print(f"{formatted_key} = {value_str}")
        
        
#%%          
            
# ChatGPT
def sort_dict_by_keys(input_dict):

    sorted_keys = sorted(input_dict.keys(), key=lambda x: x.lower())
    sorted_dict = {key: input_dict[key] for key in sorted_keys}
    
    return sorted_dict

#%%
# ChatGPT
def sort_key(element):
    # Séparer l'élément en fonction des tirets et convertir le dernier élément en entier
    # Ceci permet de trier en fonction du numéro après le dernier tiret.
    return int(element.split('-')[-1])





#%%
# ChatGPT
def string_to_binary(input_string):
    binary_string = ""
    for char in input_string:
        # Convertir chaque caractère en binaire et ajouter à la chaîne résultante
        binary_char = format(ord(char), '08b')  # Convertir le caractère en binaire sur 8 bits
        binary_string += binary_char
    return binary_string

        



#%%
            
def what(input):
    print("type = {}, shape = {}".format(type(input),input.shape))
    
    
#%%


def xml2dict(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        data_dict = {}
        
        for elem in root.iter():
            if elem.text is not None:
                try:
                    value = int(elem.text)
                except ValueError:
                    value = elem.text
                data_dict[elem.tag] = value
        
        return data_dict
    
    except ET.ParseError as e:
        print(f"Erreur de parsing XML : {e}")
        return None

