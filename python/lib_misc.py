# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 2.1.6
#   Date            : 2024-11-13
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
# 
# ----- CHANGELOG -----
#   1.0.0 (2023-03-04)  creation
#   1.1.1 (2024-04-01)  [NEW] string_to_binary / binary_to_decimal
#   1.1.2 (2024-04-03)  [NEW] plot_1y_axes
#   1.2.0 (2024-05-22)  [NEW] convert_byte, get_total_size
#   1.2.1 (2024-05-28)  create_xml_file
#   1.2.2 (2024-06-06)  Ntaps -> NsampTaps, create_xml_file
#                       KEYS, sorting and giving number of keys
#   1.3.0 (2024-06-28)  [NEW]   remove_n_characters_from_filenames,
#                       [NEW]   replace_string_in_filenames
#                       [NEW]   truncate_lr_in_filename
#                       [NEW]   add_string_in_filemane
#                       move_files_to_folder: year as argument
#                       set_saving_params: no temporary folder anymore
#                       create_xml_file: adding the realisation number to the
#                       file name
#   1.3.1 (2024-07-10)  naming normalisation (*frame*-> *Frame*).
#                           along with main (1.4.3)
#                       save2mat: adding fields for matlab processing
#                           along with main_0_python2matlab.m (2.0.2)
#   1.4.0 (2024-07-11)  [NEW]   find_string_in_list
#                       create_xml_file: managing phase noise
#                       replace_string_in_filenames, truncate_lr_in_filename
#                           allowing filenames as input instead of only paths
#                       plot_3y_axes: semilogy if SER as entry
#                       organise_files: managing xml files
#                       save2mat: exporting phase noise esitmation with cpr
#                       find_string -> find_substring
# ---------------------
#   2.0.0 (2024-07-12) - LIBRARY NAME CHANGED: LIB_GENERAL -> LIB_PLOT
#   2.0.1 (2024-07-13) - save2mat: adding 'flag_phase_noise' in output dict
#   2.0.2 (2024-07-16) - save2mat: adding rx mode
#                      - organise_files: managing phase noise 
#   2.0.3 (2024-07-24) - init_dict: server mode
#                        create_xml_file: Th_std -> fpol for filenameing
#   2.1.0 (2024-09-11) - [NEW] rename_files
#                      - move_files_to_folder: adding optional argument for
#                           moving to custom folder
#                      - remove_n_characters_from_filenames: do not remove
#                           characters if it is not a date
#   2.1.1 (2024-10-07) - save2mat: adding VSOP
#                        create_xml_file: fpol -> vsop; Sph -> CFO filenameing
#   2.1.2 (2024-10-09) - create_xml_file: vsop rounding
#   2.1.3 (2024-10-11) - create_xml_file: vsop/CFO rounding + creation or not
#                           xml file
#   2.1.4 (2024-11-05) - save2mat: adding vsop and SoPLin
#                      - init_dict: saving["server"] corrected
#                      - tx['flag_phase_noise] -> tx['flag_phase]
#   2.1.5 (2024-11-10) - create_xml_file: vsop/CFO rounding
#   2.1.6 (2024-11-13) - create_xml_file: dnu rounding
#   2.1.7 (2024-11-16) - organise_files: xml and log files management
#
# ----- MAIN IDEA -----
#   Miscellaneous functions for logistics
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
#   [C1] Author         : CNRS - IDRIS
#       Contact         : 
#       Affiliation     : 
#       Date            : 2023-05-09
#       Program Title   : Profilage de codes python
#       Code Version    : 
#       Type            : tutorial
#       Web Address     : http://www.idris.fr/jean-zay/cpu/jean-zay-cpu-python.html
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
import re

import lib_plot as plot
import lib_matlab as mb
pi = np.pi


#%% ============================================================================
# --- CONTENTS ---
# - add_string_in_filemane              (1.3.0)
# - are_same_dicts
# - are_same_tensors
# - array2csv
# - binary_to_decimal                   (1.1.1)
# - create_xml_file
# - convert_byte                        (1.2.0)
# - extract_values_from_filename
# - find_substring
# - find_string_in_list                 (1.4.0)
# - get_total_size                      (1.2.0)
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
# - plot_1y_axes                        (1.1.2)
# - plot_2y_axes
# - plot_3y_axes
# - remove_using_index
# - remove_n_characters_from_filenames  (1.3.0)
# - rename_files                        (2.1.0)
# - replace_string_in_filenames         (1.3.0)
# - save2mat
# - select_and_read_files
# - set_saving_params
# - show_dict
# - sort_dict_by_keys
# - sort_key
# - string_to_binary                    (1.1.1)
# - truncate_lr_in_filename             (1.3.0)
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

def add_string_in_filemane(path, loc, string,*varargin):

    if not os.path.isdir(path):
        print("The specified path is not a valid directory.")
        return
    

    for filename in os.listdir(path):
        flag_do = 0
        if len(varargin) == 0:
            if os.path.isfile(os.path.join(path, filename)):
            
                flag_do = 1

        else:
            if len(varargin) == 1 and varargin[0] in filename.lower():
                flag_do = 1
                
            if len(varargin) == 2\
                and varargin[0] in filename.lower()\
                and varargin[1] not in filename.lower():
                
                    flag_do = 1

        if flag_do:
            new_filename    = filename[:loc] + string + filename[loc:]
            old_file        = os.path.join(path, filename)
            new_file        = os.path.join(path, new_filename)
        
            os.rename(old_file, new_file)
    

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
def create_xml_file(tx,fibre,rx,saving,gen,*varargin):
    
    sections    = ["TX","CHANNEL","RX"]


# =============================================================================
# default values in the filenames
# =============================================================================
    if tx['nu'] != 0:
        TX      = ["mod","nu","Nsps", "Rs","NsampTaps",'SNRdB']
    else:
        TX      = ["mod", "Nsps", "Rs","NsampTaps",'SNRdB']
        
    if tx['flag_phase']:
        if tx['PhiLaw']["kind"] == "Rwalk":
            TX.append('dnu')
        else:
            TX.append('law')

    CHANNEL     = ["tauPMD", "tauCD", "law"]
    RX          = ["mimo",'lr',"NFrames", "NSymbBatch", "FrameChannel", "NSymbFrame","SNR_dB"]
    fields_list = [TX, CHANNEL, RX]
    
    if tx['nu'] != 0:
        saving_list = ["mimo",'lr','Rs','mod',"nu",'SNRdB',"CD","PMD",'Thlaw',"NSymbBatch","NSymbFrame","SNR_dB","NsampTaps"]
        if tx['PhiLaw']["kind"] == "Rwalk":
            saving_list.insert(5,"dnu")
    else:
        saving_list = ["mimo",'lr','Rs','mod','SNRdB',"CD","PMD",'Thlaw',"NSymbBatch","NSymbFrame","SNR_dB","NsampTaps"]
        if tx['PhiLaw']["kind"] == "Rwalk":
            saving_list.insert(4,"dnu")        

    CHANNELpar  = [np.round(fibre["tauPMD"]*1e12,0),           # [ps]
                   np.round(np.sqrt(fibre["tauCD"])*1e12,0),   # [ps]
                   fibre["ThetasLaw"]["law"]]
    
# =============================================================================
# additional values in the filenames
# =============================================================================
    
    if fibre["ThetasLaw"]["kind"] == "Rwalk":
        
        if fibre["ThetasLaw"]["law"] == "uni":
            saving_list.insert(6,'low')
            saving_list.insert(7,'high')
            
            CHANNEL.append('low')
            CHANNEL.append('high')
            
            CHANNELpar.append(fibre["ThetasLaw"]['low']*180/pi)
            CHANNELpar.append(fibre["ThetasLaw"]['high']*180/pi)
        
        if fibre["ThetasLaw"]["law"] == "gauss":
            saving_list.insert(6,'Th_in')
            saving_list.insert(7,'vsop')
            
            CHANNEL.append('Th_in')
            CHANNEL.append('vsop')
            
            CHANNELpar.append(np.round(fibre["ThetasLaw"]['theta_in']*180/np.pi,0))
            CHANNELpar.append(int(fibre['vsop']*1e-4))    # [10x krad/s]
            
        
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
            saving_list.insert(11,'ThEnd')
            saving_list.insert(12,'Sth')

            CHANNEL.append('ThEnd')
            CHANNEL.append('Sth')
            
            Sth = (fibre["ThetasLaw"]['End']-fibre["ThetasLaw"]['Start'])/(rx['NFrames']-rx['FrameChannel']) # [rad]
            CHANNELpar.append(int(fibre["ThetasLaw"]['End']*180/pi))
            CHANNELpar.append(np.round(Sth*180/pi,2))
            


    TXpar       = [tx["mod"],
                    tx["Nsps"],
                    int(tx["Rs"]*1e-9),
                    tx['NsampTaps'],
                    tx['SNRdB'],
                    round(tx["dnu"]*1e-4,1) if tx['PhiLaw']["kind"] == "Rwalk"\
                    else tx['PhiLaw']["law"],
                    ]
    
    if tx["nu"] != 0:
        TXpar.insert(1, round(tx['nu'],3))

    if tx['flag_phase'] == 1:
        if tx["PhiLaw"]["kind"] == "func":
            if tx["PhiLaw"]["law"] == "lin":
                saving_list.insert(6,'PhEnd')
                saving_list.insert(7,'CFO')
    
                TX.append('PhEnd')
                TX.append('CFO')
                
                Sph = (tx["PhiLaw"]['End']-tx["PhiLaw"]['Start'])/(rx['NFrames']-rx['FrameChannel']) # [rad]
                TXpar.append(int(tx["PhiLaw"]['End']*180/pi))
                TXpar.append(int(tx["PhiLaw"]['CFO']*1e-4))
    else:
        TXpar = TXpar[:-1]
    
    
    if rx['mode'].lower() == 'pilots':
        saving_list.insert(len(TX), "Nplts")
        TX.append("Nplts")
        TXpar.insert(len(TXpar), tx['pilots_info'][0][-1])

    
    RXpar       = [rx['mimo'],
                   round(rx['lr']*1000,4) if rx['mimo'].lower() == 'vae'
                       else round(rx['lr']*1e6,2),
                   rx["NFrames"],
                   rx["NSymbBatch"],
                   rx["FrameChannel"],
                   np.round(rx["NSymbFrame"]*1e-3,2),
                   rx['SNRdB']]
    
    params_list             = [TXpar, CHANNELpar,RXpar]
    [base_path, my_path]    = set_saving_params()
    
    
    os.chdir(base_path)
    
    if gen:
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
    filename    = datetime.now().strftime("%y-%m-%d %H:%M:%S")
    
    
    if len(varargin) != 0:
        filename  += f'   {varargin[0]}'
        
    for name in saving_list:
        # print("\n"*3)
        # print(name)
 
        # find the index of the field in fields_list
        for section_fields in fields_list:
            # print(section_fields)
            
            if name in section_fields:
                index = section_fields.index(name)
            
                # Uses the index to get the value corresponding to the field in params_lists
                value = params_list[fields_list.index(section_fields)][index]
                # print(value)
                filename += f" - {name} {value}"

    filename += ".xml"


    filename = replace_string_in_filenames(filename, "NSymbBatch","NSbB")
    filename = replace_string_in_filenames(filename, "NSymbFrame","NSbF")
    filename = replace_string_in_filenames(filename, "NsampTaps","NspT")
    filename = truncate_lr_in_filename(filename)
    

    if gen:
        # create the xml file
        tree = ET.ElementTree(root)
        tree.write(filename, encoding="utf-8", xml_declaration=True)
    
    os.chdir("../")

    # print(file_name[:-4])
    return filename[:-4]





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
def find_string_in_list(string,mylist):
    
    assert string in mylist, f"{string} not in the list"
    
    indexes = []
    k       = 0
    
    while k < len(mylist):
        if string == mylist[k]:
            indexes.append(k)
        k += 1
        
    return indexes

#%%

def find_substring(pattern, mystring):
    
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
            
    plot.constellations(data['tx_real'],"tx")
    plot.constellations(data['rx_real'],"rx")

    return data
        


#%%            
def init_dict(server):
    
    tx          = dict()
    fibre       = dict()
    rx          = dict()
    saving      = dict()

    fibre['ThetasLaw']  = dict()
    tx["PhiLaw"]        = dict()

    for field_name in ["mod","Nsps","Rs","nu","NsampTaps"]:
        tx[field_name] = 0
        
    #for field_name in ["channel",'PMD','CD','phiIQ','theta1','theta_std']:
    for field_name in ['PMD','CD','phiIQ']:
        fibre[field_name] = 0
    
        
    for field_name in ['SNRdB',"NSymbBatch","N_lrhalf","NFrames","FrameChannel","Nmax_SymbFrame","mimo"]:
        rx[field_name] = 0
        
    rx["Frame"]             = 0
    tx["RollOff"]           = 0.1                   # roll-off factor
    tx["Nsps"]              = 2                     # oversampling factor (Shannon Coefficient) in samples per symbol
    tx["nu"]                = 0                     # [0] [0.0270955] [0.0872449] [0.1222578]
    rx["N_lrhalf"]          = 170

    saving['root_path'] = mb.PWD(show = False)
    saving['merge_path']= saving['root_path']+'/data-'+str(date.today())

    if server:
        tx["server"]        = 1
        rx['server']        = 1
        saving['server']    = 0
    else:
        tx["server"]        = 0
        rx['server']        = 0
        saving['server']    = 1        

    tx      = sort_dict_by_keys(tx)
    fibre   = sort_dict_by_keys(fibre)
    rx      = sort_dict_by_keys(rx)
    saving  = sort_dict_by_keys(saving)

    return tx,fibre,rx,saving

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
    dict = sort_dict_by_keys(dict)
    
    for key in dict.keys():
        print(key)
        
    print('\n number of keys = {}'.format(len(dict.keys())))

    
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
        # list of all the files and repositories
        folders = os.listdir(path)

        # select only folders
        folders = [folder for folder in folders\
                   if os.path.isdir(os.path.join(path, folder))]

        return folders
    
    except OSError as e:
        print(f"Error : {e}")
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
   
    folders_to_delete   = []
    folders             = sorted(folders_tmp,key=sort_key)
    Nfolders            = len(folders)
    merge_folder        = saving["merge_path"][-(4+4+2+2+3):]
    # 4 for 'data', 4 for '2024', 2 for 'yy', 'mm', 'dd', 3*1 for '-'
    
    for k in range(Nfolders):
        if (merge_folder in folders[k]) & (merge_folder != folders[k]):
            folders_to_delete.append(folders[k])

    folders_to_delete = sorted(folders_to_delete,key=sort_key)

    for folder in folders_to_delete:
        source_folder = os.path.join(saving['root_path'], folder)
        
        for file_name in os.listdir(source_folder):
            # Vérifier si le fichier a la même date que le dossier de fusion
            # 4 for 'data', 1 for '-', 2 for '20'
            merge_folder_tmp = merge_folder[4+1+2:]
            
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
def move_files_to_folder(year, target_folder=None):
    # Obtenez la liste des fichiers dans le répertoire actuel
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    
    for filename in files:
        # Vérifiez si le nom du fichier commence par l'année spécifiée
        if filename.startswith(str(year).zfill(2)):
            if target_folder:
                # Vérifiez si le dossier cible existe, sinon, créez-le
                if not os.path.exists(target_folder):
                    os.mkdir(target_folder)
                # Déplacez le fichier dans le dossier spécifié
                shutil.move(filename, os.path.join(target_folder, filename))
            else:
                # Comportement original basé sur la date dans le nom du fichier
                parts       = filename.split(' - ')
                date_part   = parts[0].split(' ')
                day_part    = date_part[0]

                if is_date(day_part):
                    date_folder = f'data-{day_part}'

                    # Vérifiez si le dossier existe déjà, sinon, créez-le
                    if not os.path.exists(date_folder):
                        os.mkdir(date_folder)
                    # Déplacez le fichier dans le dossier approprié
                    shutil.move(filename, os.path.join(date_folder, filename))



      
#%%

def my_tensor(size,device='cpu',dtype=torch.float32,requires_grad=False):
    return torch.tensor(size,device=device,\
                        dtype=dtype,requires_grad=requires_grad)



#%%

def my_zeros_tensor(size,device='cpu',dtype=torch.float32,requires_grad=False):
    return torch.zeros(size,device=device,\
                       dtype=dtype,requires_grad=requires_grad)


#%%
def organise_files(directory,gen_xml):
    # Define the main target directories
    figs_dir    = os.path.join(directory, 'figs')
    mat_dir     = os.path.join(directory, 'mat')
    log_dir     = os.path.join(directory, 'log')
    csv_dir     = os.path.join(directory, 'csv')
    err_dir     = os.path.join(directory, 'err')
    err_dir_bt  = os.path.join(err_dir, 'betas')
    err_dir_th  = os.path.join(err_dir, 'thetas')
    err_dir_ph  = os.path.join(err_dir, 'phis')
    if gen_xml:
        xml_dir     = os.path.join(directory, 'xml')
        
    # Create the main target directories if they don't exist
    os.makedirs(figs_dir,   exist_ok=True)
    os.makedirs(mat_dir,    exist_ok=True)
    os.makedirs(csv_dir,    exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(err_dir,    exist_ok=True)
    os.makedirs(err_dir_bt, exist_ok=True)
    os.makedirs(err_dir_th, exist_ok=True)
    os.makedirs(err_dir_ph, exist_ok=True)
    if gen_xml:
        os.makedirs(xml_dir,    exist_ok=True)


    
    # Create subdirectories for images
    python_dir      = os.path.join(figs_dir, 'python')
    poincare_dir    = os.path.join(figs_dir, 'poincare')
    fir_dir         = os.path.join(figs_dir, 'fir')

    os.makedirs(python_dir  , exist_ok=True)
    os.makedirs(poincare_dir, exist_ok=True)
    os.makedirs(fir_dir     , exist_ok=True)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # check it is a file and not a folder
        if os.path.isfile(filepath):
            if filename.endswith('.svg'):
                shutil.move(filepath, os.path.join(python_dir, filename))

            elif filename.endswith('.png'):

                if 'poincare' in filename.lower():
                    shutil.move(filepath, os.path.join(poincare_dir, filename))
                else:
                    shutil.move(filepath, os.path.join(fir_dir, filename))

            elif filename.endswith('.mat'):
                shutil.move(filepath, os.path.join(mat_dir, filename))

            elif filename.endswith('.xml'):
                shutil.move(filepath, os.path.join(xml_dir, filename))
                
            elif filename.endswith('.log'):
                shutil.move(filepath, os.path.join(log_dir, filename))

            elif filename.endswith('.csv'):
                if 'err_bt' in filename.lower():
                    shutil.move(filepath, os.path.join(err_dir_bt, filename))
                elif 'err_th' in filename.lower():
                    shutil.move(filepath, os.path.join(err_dir_th, filename))
                elif 'err_ph' in filename.lower():
                    shutil.move(filepath, os.path.join(err_dir_ph, filename))
                else:
                    shutil.move(filepath, os.path.join(csv_dir, filename))

            else:
                print(f'Unrecognized file type: {filename}')


#%%
        
# https://www.kite.com/python/answers/how-to-remove-an-element-from-an-array-in-python
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
def remove_n_characters_from_filenames(directory, n,*varargin):

    for filename in os.listdir(directory):

        if os.path.isfile(os.path.join(directory, filename)):
            if len(varargin) != 0 and filename[0].isnumeric() == False:
                continue
            else:
                new_filename = filename[n:]
                os.rename(
                    os.path.join(directory, filename),
                    os.path.join(directory, new_filename)
                )

#%% ChatGPT
def replace_string_in_filenames(string, old_string, new_string):

    if not os.path.isdir(string):
        what = "filename"
    else:
        what = "path"
    
    if what == "path":
        for root, dirs, files in os.walk(string):
            for filename in files:
    
                if old_string in filename:
    
                    new_filename = filename.replace(old_string, new_string)
                    old_filepath = os.path.join(root, filename)
                    new_filepath = os.path.join(root, new_filename)
                    
                    os.rename(old_filepath, new_filepath)
    else:
        new_filename = string.replace(old_string, new_string)
        return new_filename
        
#%%
def rename_files(path, string, position):
    # Check if the directory exists
    if not os.path.isdir(path):
        raise ValueError(f"The path {path} is not a valid directory.")
    
    count = 0
    # Iterate over all files in the directory
    for filename in os.listdir(path):
        str_len = 4 # 4 == ma g/u
        
        if filename[0:str_len] != string[0:str_len]:
            # Split the file name and its extension
            file_base, file_ext = os.path.splitext(filename)
            
            # If the position is valid, insert the string at the given position
            if 0 <= position <= len(file_base):
                new_name = file_base[:position] + string + file_base[position:] + file_ext
            else:
                raise ValueError(f"The position {position} exceeds the length of the file name {file_base}.")
            
            # Create the full paths for the old and new file names
            old_path = os.path.join(path, filename)
            new_path = os.path.join(path, new_name)
            count = count + 1
            
            # Rename the file
            os.rename(old_path, new_path)
    print(f"{count} files renamed")
        
#%%
def rename_files2(path, string, position):
    # Check if the directory exists
    if not os.path.isdir(path):
        raise ValueError(f"The path {path} is not a valid directory.")
    
    count = 0
    # Iterate over all items in the directory
    for filename in os.listdir(path):
        # Create the full path for the item
        full_path = os.path.join(path, filename)
        
        # Skip directories
        if os.path.isdir(full_path):
            continue
        
        str_len = 4 # 4 == ma g/u
        
        # Check if the file name starts with the specified string
        if filename[0:str_len] != string[0:str_len]:
            # Split the file name and its extension
            file_base, file_ext = os.path.splitext(filename)
            
            # If the position is valid, insert the string at the given position
            if 0 <= position <= len(file_base):
                new_name = file_base[:position] + string + file_base[position:] + file_ext
            else:
                raise ValueError(f"The position {position} exceeds the length of the file name {file_base}.")
            
            # Create the full paths for the old and new file names
            new_path = os.path.join(path, new_name)
            count += 1
            
            # Rename the file
            os.rename(full_path, new_path)
    
    print(f"{count} files renamed")
#%%
def save2mat(tx,fibre,rx,saving):


    name        = saving["filename"]+".mat"
    
    save_dict   = {
                'NspTaps'           : tx['NsampTaps'],
                'nu'                : tx["nu"],
                'nu_sc'             : tx["nu_sc"],
                'Rs'                : tx["Rs"],
                'thetas'            : np.array(fibre["thetas"]), # tensor type not recognised by matlab
                'NSymbBatch'        : rx["NSymbBatch"],
                'h_gnd_frame'       : rx["h_gnd"] if rx['save_channel_gnd'] else np.nan,
                'h_est_frame'       : rx["h_est_frame"],
                'h_est_batch'       : rx["h_est_batch"] if rx['save_channel_batch'] else np.nan,
                'lr'                : rx["lr"],
                'Losses'            : rx["Losses"],
                'NFrames'           : rx["NFrames"],
                'NFramesTraining'   : rx["NFramesTraining"],
                'FrameChannel'      : rx["FrameChannel"],
                'NBatchesFrame'     : rx['NBatchFrame'],
                'NBatchesChannel'   : rx['NBatchesChannel'],
                'Phis_gnd'          : tx["PhaseNoise_unique"] if tx['flag_phase'] else np.nan,
                'SNRdBs'            : rx["SNRdBs"],
                'SER_means'         : rx["SERs"],
                'SER_valid'         : rx["SER_valid"],
                'Var_est'           : rx["Pnoise_est"],
                'flag_phase_noise'  : tx['flag_phase'],
                'PhaseNoise_est_cpr': rx['PhaseNoise_pilots'] if rx['mode'].lower() == "pilots"\
                                            else np.nan,
                'ThLaw'             : fibre["ThetasLaw"]['law'],
                'PhLaw'             : tx['PhiLaw']["law"],
                'rx_mimo'           : rx['mimo'],
                'rx_mode'           : rx['mode'],
                'vsop'              : fibre['vsop'] if fibre['ThetasLaw']['kind'].lower() == 'rwalk'\
                                        else np.nan,
                'SoPlin'            : fibre['ThetasLaw']["SoPlin"] if fibre['ThetasLaw']['kind'].lower() == 'func'\
                                        else np.nan,
                }
    
    
    if rx['mimo'].lower() == "vae":
        save_dict['Var_real']   = rx["noise_var"].detach().numpy()

    else:
        save_dict['Var_real']   = rx["noise_var"]
        
        
    if "thetadiffs" in fibre:
        del fibre['thetadiffs']
        
    io.savemat(name,save_dict)



    
#%%

def select_and_read_files(extension="csv",skiprow = 0,\
                          title = "Select files",Multiple = True):
    """
    return[0] = data_matrices
    return[1] = field_names
    return[2] = file_names
    return[3] = file_paths
    """
    
    root = tk.Tk()
    root.withdraw() # hide tkinter window

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
            
        file_name = file_path.split("/")[-1]
        file_names.append(file_name)

        if extension == "csv":
            data = np.loadtxt(file_path, delimiter=",", skiprows=skiprow)
            data_matrices.append(data)

    if skiprow == 0:
        return data_matrices,file_names, file_paths
    
    else:
        return data_matrices, field_names, file_names, file_paths
    
    


#%%
            
def set_saving_params():
    
    current_day = date.today().strftime("%y-%m-%d")

    base_path   = "data-"+current_day
    k           = 0
    my_path_tmp = base_path

    try:
        os.mkdir(base_path)
    except:
        pass

    return [base_path, my_path_tmp]




#%%

# ChatGPT
def show_dict(my_dict):
    
    my_dict = sort_dict_by_keys(my_dict)
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
    # split the element according to the dashes and convert the last one as an
    # integer which enables sorting by the dash number

    return int(element.split('-')[-1])



#%%
# ChatGPT
def string_to_binary(input_string):
    binary_string = ""
    for char in input_string:
        #  convert each character in 8bits binary and adding it to the existing
        #  string
    
        binary_char     = format(ord(char), '08b')
        binary_string += binary_char
    return binary_string


#%% ChatGPT
def truncate_lr_in_filename(string):
    if not os.path.isdir(string):
        what = "filename"
    else:
        what = "path"

    lr_pattern = re.compile(r'(lr\s)(\d+\.\d+)')
    
    if what == 'path':
        for filename in os.listdir(string):
            
            # Search for the pattern in the filename
            match = lr_pattern.search(filename)
            
            if match:
                lr              = float(match.group(2))
                lr_trunc        = f'{lr:.2f}'
                new_filename    = filename.replace(match.group(2), lr_trunc)            
                old_file        = os.path.join(string, filename)
                new_file        = os.path.join(string, new_filename)
                
                os.rename(old_file, new_file)
    else:
        match = lr_pattern.search(string)
        if match:
            lr              = float(match.group(2))
            lr_trunc        = f'{lr:.2f}'
            new_filename    = string.replace(match.group(2), lr_trunc)

        return new_filename

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
        print(f"XML parsing error: {e}")
        return None

