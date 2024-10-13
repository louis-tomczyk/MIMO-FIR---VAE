import os
import argparse

def extract_version_and_date(file_path):
    try:
        with open(file_path, 'r') as file:
            # Lire toutes les lignes du fichier
            lines = file.readlines()
            
            # S'assurer que le fichier a au moins 7 lignes
            if len(lines) < 7:
                return "Le fichier ne contient pas suffisamment de lignes", None
            
            # Extraire la version (6ème ligne)
            version_line = lines[5].strip()
            version = version_line.split(':')[1].strip()
            
            # Extraire la date (7ème ligne)
            date_line = lines[6].strip()
            date = date_line.split(':')[1].strip()
            
            return version, date
    except FileNotFoundError:
        return "Le fichier n'existe pas", None
    except Exception as e:
        return f"Erreur : {e}", None

def process_files_in_directory(directory_path, extensions):
    results = {}
    # Liste des fichiers avec les extensions spécifiées, triés par ordre alphabétique
    files = sorted([filename for filename in os.listdir(directory_path) if any(filename.endswith(ext) for ext in extensions)])
    for filename in files:
        file_path = os.path.join(directory_path, filename)
        version, date = extract_version_and_date(file_path)
        results[filename] = (version, date)
    return results

# Exemple d'utilisation
def example_usage(what, extensions):
    if os.path.isfile(what):
        version, date = extract_version_and_date(what)
        print(f"version: {version} --- Date : {date} --- {what}")
    else:
        results = process_files_in_directory(what, extensions)
        for filename, (version, date) in results.items():
            print(f"version: {version} --- Date : {date} --- {filename}")

# Appeler l'exemple d'utilisation
what        = "/home/louis/Documents/6_Telecom_Paris/3_Codes/0_louis/2_VAE/matlab/lib" #os.getcwd()
# what        = os.getcwd()
extensions  = '.mat'#['.py', '.mat', '.tex']
example_usage(what, extensions)