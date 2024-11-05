# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.0.0
#   Date            : 2024-10-14
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2023-03-04) - creation
# 
# ----- MAIN IDEA -----
#   merge log files 
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books
#   [A1] Authors         : 
#        Title           :
#        Journal/Editor  :
#        Volume - N°     :
#        Date            :
#        DOI/ISBN        :
#        Pages           :
#  ----------------------
#   CODE
#   [C1] Author          : Pratik Deoghare
#        Contact         : pratik.deoghare@gmail.com
#        Laboratory/team :
#        Institution     :
#        Date            : 2014-07-23
#        Program Title   : 
#        Code Version    : 
#        Web Address     : https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring
# ---------------------------------------------
# %%


#%% ===========================================================================
# --- CONTENTS ---
# =============================================================================
# - find_all
# - merge_logs
# - correct_merged_log
# - convert_log_to_csv
# - sort_csv
# - remove_columns
# - rename_file
# - delete_tmp_files
# =============================================================================


#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================

import os
import csv

#%% ===========================================================================
# --- PARAMETERS ---
# =============================================================================

'''
Merged      = '0_tmp_merged_fail.log'
Corrected   = '1_tmp_corrected_fail.log'
Converted   = '2_tmp_converted_file.csv'
Sorted      = '3_tmp_sorted_fail.csv'
Removed     = '4_tmp_removed_col.csv'
Renamed     = 'logs.csv'

log_dir = '/home/louis/Documents/6_TélécomParis/3_Codes/0_louis/2_VAE/python'
'''

#%%
# [C1]
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)

#%%
def merge_logs(directory, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(directory):
            if filename.endswith('.log'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as infile:
                    content = infile.read().strip()
                    outfile.write(content + '\n')

#%%
def correct_merged_log(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        new_lines = []  # Liste pour stocker les nouvelles lignes du fichier
        lines_to_move = []  # Liste pour stocker les lignes qui seront déplacées à la fin
        
        for line in infile:
            n_failed = line.count('failed')
            if n_failed > 1:
                failed_locs = list(find_all(line, 'failed'))
                
                # Garder uniquement la première occurrence dans la ligne actuelle
                first_part = line[:failed_locs[1]].strip()
                new_lines.append(first_part + '\n')

                # Déplacer les autres occurrences à la fin
                for loc in failed_locs[1:]:
                    rest_part = line[loc:].strip()
                    lines_to_move.append(rest_part + '\n')
            else:
                new_lines.append(line)  # Si une seule occurrence, on garde la ligne telle quelle
        
        # Écrire les lignes normales
        outfile.writelines(new_lines)
        # Ajouter les lignes déplacées à la fin
        outfile.writelines(lines_to_move)

#%%
def convert_log_to_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Supprimer les espaces inutiles et remplacer les "=" et " " par des virgules
            csv_line = line.replace(' = ', ',').replace(' - ', ',').replace('[', '').replace(']', '').replace(' ',',')
            outfile.write(csv_line)
            
#%%
def sort_csv(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        rows = list(reader)  # Convertir les lignes en liste

        # Trier par SNR (colonne 2), puis Nrea (colonne 4), puis NSbB (colonne 6), etc.
        sorted_rows = sorted(rows, key=lambda row: \
                     (float(row[3]), int(row[5]), float(row[9]), 
                      int(row[11]), int(row[13]),
                      int(row[11]), int(row[13]), int(row[15]))
                     )

    # Écrire les lignes triées dans un nouveau fichier CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(sorted_rows)
        
#%%
def remove_columns(input_file, output_file, col):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            new_row = [val for i, val in enumerate(row) if i not in col]
            writer.writerow(new_row)

#%%
def rename_file(log_dir, input_file, output_file):
    
    current_dir     = os.getcwd()
    os.chdir(log_dir)
    first_log_file  = os.listdir(log_dir)[0]
    date_part       = first_log_file.split()[0]
    

    new_output_name = f"/{date_part}_{output_file}"
    os.chdir(current_dir)
    
    input_file_path = os.path.join(os.getcwd(), input_file)
    os.rename(input_file_path, current_dir+new_output_name)

#%%
def delete_tmp_files(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)
        else:
            print(f"file {file} does not exists")


'''
#%% ===========================================================================
# --- MAIN ---
# =============================================================================
     

merge_logs(log_dir, Merged)
correct_merged_log(Merged, Corrected)
convert_log_to_csv(Corrected, Converted)
sort_csv(Converted, Sorted)
remove_columns(Sorted, Removed, [0])
rename_file(log_dir, Removed, Renamed)
delete_tmp_files([Merged, Corrected, Converted, Sorted])



'''