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
#   Authors             : Vincent LAUINGER
#   Title               : Blind Equalization and Channel Estimation in Coherent Optical
#                         Communications Using Variational Autoencoders
#   Jounal/Editor       : JOURNAL ON SELECTED AREAS IN COMMUNICATIONS
#   Volume - N°         : 40-9
#   Date                : 2022-11
#   DOI/ISBN            : 10.1109/JSAC.2022.3191346
#   Pages               : 2529 - 2539
#  ----------------------
#   Functions           : 
#   Author              : Vincent LAUINGER
#   Author contact      : vincent.lauinger@kit.edu
#   Affiliation         : Communications Engineering Lab (CEL)
#                           Karlsruhe Institute of Technology (KIT)
#   Date                : 2022-06-15
#   Title of program    : 
#   Code version        : 
#   Type                : source code
#   Web Address         :
# ---------------------------------------------
# %%



# ========================================================================================== #
# MAINTENANCE
# ============================================================================================ #
import lib_matlab as matlab
matlab.clear_all()
import numpy as np
import torch
import lib_misc as misc
import processing_self as process

parallelisation = True
if parallelisation == True:
    from joblib import Parallel,delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda": torch.cuda.init()
torch.set_default_dtype(torch.float32)


#%%
# ============================================================================================ #
# SIMULATION PARAMETERS
# ============================================================================================ #

tx,fibre,rx,saving,flags= misc.init_dict()

# ------- TX
tx["mode"]              = "self"                # {'self','ext'}
tx["mod"]               = '4QAM'                # modulation format:  {4,16,64}-QAM
tx["Nsps"]              = 2                     # oversampling factor (Shannon Coefficient) in samples per symbol
tx["Rs"]                = 40e9                  # Symbol rate [Baud]
tx["nu"]                = 0                     # [0] [0.0270955] [0.0872449] [0.1222578]
tx["RoffOff"]           = 0.1                   # roll-off factor
tx["NSymbTaps"]         = 7                     # >6
tx["SNRdB"]             = 90

# ------- RX
rx["SNRdB"]             = 25
rx["NBatchFrame"]       = 100
rx["NSymbFrame"]        = 2000


rx["Nframes"]           = 40
rx["FrameRndRot"]       = 20
rx["lr"]                = 3e-4

# ------- FIBRE
fibre['TauPMD']                 = 0*1e-24*np.sqrt(1000) # pol. mode dispersion induced delay [s]
fibre['TauCD']                  = 0*1e-24               # differential group delay [ps]
fibre['phiIQ']                  = np.array([0+0j,0+0j]) # static IQ-shift [rad]
fibre["ThetasLaw"]['theta_in']  = 0                     # HV shift [rad]


# fibre['ThetasLaw']["kind"]     = 'func'
# fibre['ThetasLaw']["law"]      = 'lin'
# fibre["ThetasLaw"]['Start']    = 0
# fibre["ThetasLaw"]['End']      = np.pi/100
# fibre["ThetasLaw"]['Slope']    = (fibre["ThetasLaw"]['End']-fibre["ThetasLaw"]['Start'])/rx["Nframes"]



# fibre['ThetasLaw']["kind"]          = 'Rwalk'
# fibre["ThetasLaw"]['law']           = 'uni'
# fibre["ThetasLaw"]['low']           = -np.pi/100
# fibre["ThetasLaw"]['high']          = np.pi/100


fibre['ThetasLaw']["kind"]          = 'Rwalk'
fibre["ThetasLaw"]['law']           = 'gauss'
fibre["ThetasLaw"]['theta_in']      = 0
fibre["ThetasLaw"]['theta_std']     = np.pi/30

# fibre['ThetasLaw']["kind"]          = 'Rwalk'
# fibre["ThetasLaw"]['law']           = 'expo'
# fibre["ThetasLaw"]['mean']          = 1

# fibre['ThetasLaw']["kind"]          = 'Rwalk'
# fibre["ThetasLaw"]['law']           = 'tri'
# fibre["ThetasLaw"]['low']           = -np.pi/25
# fibre["ThetasLaw"]['high']          = np.pi/25
# fibre["ThetasLaw"]['mode']          = 0





Nrea            = 1
param           = \
    list(np.linspace(1,10,10)*1e3)+\
    list(np.linspace(20,100,9)*1e3)    
    
flags["plot_const_batch"]   = False
flags["plot_loss_batch"]    = True
flags["save_plots"]         = True


def process_data(k,tx,fibre,rx):

    tx["linewidth"]         = param[k]
    print(tx["linewidth"])
    saving["filename"]      = misc.create_xml_file(tx,fibre,rx,saving)[2:-4]
    tx,fibre,rx             = process.processing_self(tx,fibre,rx,saving,flags)
    
    misc.save2mat(tx,fibre,rx,saving)
    misc.plot_3y_axes(saving,"iteration",'loss','SNR','Thetas',['svg'])

    print("\n"*3)
    return tx,fibre,rx

#%%
# ============================================================================================ #
# LAUNCHING and SAVING
# ============================================================================================ #

for k in range(Nrea):
    Nparams = len(param)
    
    for m in range(Nparams):
        
        if parallelisation == True:
            results     = Parallel(n_jobs=2)(
                delayed(process_data)(k,tx,fibre,rx) for k in range(Nparams))
            
        else:
            tx,fibre,rx     = process_data(m,tx,fibre,rx)

#%%
misc.move_files_to_folder()
misc.merge_data_folders(saving)




















####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################









#%%
# ============================================================================================ #
# device selection and setting saving path
# ============================================================================================ #
def init_main():
    from datetime import date
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda": torch.cuda.init()
    torch.set_default_dtype(torch.float32)
    
    saving              = dict()
    saving['root_path'] = misc.PWD()
    saving['merge_path']= saving['root_path']+'/data-'+str(date.today())
    saving              = misc.sort_dict_by_keys(saving)
    
    return device, saving





