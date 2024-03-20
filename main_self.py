# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2023-03-13 : 1.0.0 - phase noise added + cleaning
#                   : 2023-03-14 : 1.1.0 - phase/pol dynamics + cleaning
#   Date            : 2023-03-20 : 1.2.0 - CPE + phase noise
#   Version         : 1.1.0
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

parallelisation = False
if parallelisation == True:
    from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda": torch.cuda.init()
torch.set_default_dtype(torch.float32)


#%%
# ============================================================================================ #
# SIMULATION PARAMETERS
# ============================================================================================ #

tx,fibre,rx,saving,flags  = misc.init_dict()
saving['skip']          = 10

# TX
tx["mode"]              = "self"    # {'self','ext'}
tx["mod"]               = '4QAM'    # modulation format:  {4,16,64}-QAM
tx["Rs"]                = 40e9      # Symbol rate [Baud]
tx["NSymbTaps"]         = 7         # >6
tx["SNRdB"]             = 90
tx["nu"]                = 0         # [0] [0.0270955] [0.0872449] [0.1222578]

# RX
rx["SNRdB"]             = 25
rx["Nframes"]           = 100
rx["FrameRndRot"]       = 20
rx["lr"]                = 5e-4
rx["mimo"]              = "vae"     # {cma,vae}

# =============================================================================
# OLD fashion
# =============================================================================
# rx["NBatchFrame"]       = 100
# rx["NSymbFrame"]        = 3000
# rx['BatchLen']          = np.floor(rx['NSymbFrame']/rx["NBatchFrame"]).astype("int")
    
# =============================================================================
# NEW fashion
# =============================================================================
tx["dnu"]               = 1e4       # [Hz]
fibre['fpol']           = 100       # [Hz]
tx['DeltaPhiC']         = 1e-1      # [rad]
fibre['DeltaThetaC']    = np.pi/35  # [rad]
tx,fibre, rx            = misc.set_Nsymbols(tx,fibre,rx)


tx['PhiLaw']["kind"]     = 'func'
tx['PhiLaw']["law"]      = 'lin'
tx["PhiLaw"]['Start']    = 0
tx["PhiLaw"]['End']      = 2*np.sqrt(tx['Nsamp_gross']*tx['dnu']/tx['Rs'])


fibre['PMD']                    = 0*1e-24*np.sqrt(1000) # pol. mode dispersion induced delay [s]
fibre['CD']                     = 0*1e-24               # differential group delay [ps]
fibre['phiIQ']                  = np.array([0+0j,0+0j]) # static IQ-shift [rad]
fibre["ThetasLaw"]['theta_in']  = 0                     # HV shift [rad]


fibre['ThetasLaw']["kind"]     = 'func'
fibre['ThetasLaw']["law"]      = 'lin'
fibre["ThetasLaw"]['Start']    = 0
fibre["ThetasLaw"]['End']      = 70*np.pi/180



# fibre['ThetasLaw']["kind"]          = 'Rwalk'
# fibre["ThetasLaw"]['law']           = 'uni'
# fibre["ThetasLaw"]['low']           = -np.pi/100
# fibre["ThetasLaw"]['high']          = np.pi/100


# fibre['ThetasLaw']["kind"]          = 'Rwalk'
# fibre["ThetasLaw"]['law']           = 'gauss'
# fibre["ThetasLaw"]['theta_in']      = 0
# fibre["ThetasLaw"]['theta_std']     = 1*np.pi/180

# fibre['ThetasLaw']["kind"]          = 'Rwalk'
# fibre["ThetasLaw"]['law']           = 'expo'
# fibre["ThetasLaw"]['mean']          = 1

# fibre['ThetasLaw']["kind"]          = 'Rwalk'
# fibre["ThetasLaw"]['law']           = 'tri'
# fibre["ThetasLaw"]['low']           = -np.pi/25
# fibre["ThetasLaw"]['high']          = np.pi/25
# fibre["ThetasLaw"]['mode']          = 0





Nrea    = 1
param   = np.linspace(10,100,1)*1e3
flags["plot_const_batch"]   = False
flags["plot_loss_batch"]    = True
flags["save_plots"]         = True

def process_data(k,tx,fibre,rx):

    saving["filename"]      = misc.create_xml_file(tx,fibre,rx,saving)[2:-4]
    tx,fibre,rx             = process.processing_self(tx,fibre,rx,saving,flags)
    
    misc.save2mat(tx,fibre,rx,saving)
    # misc.plot_3y_axes(saving,"iteration",'loss','SNR','Thetas',['svg'])
    # misc.plot_2y_axes(saving,"iteration",'loss','Thetas',['svg'],rx['FrameRndRot'])
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
                # delayed(process_data)(k,tx,fibre,rx) for k in range(1))        
            # resuts        = list
            # len(results)  = Nparams
            # results[k]    = tuple
            # results[k][0] = tx
            # results[k][1] = fibre
            # results[k][2] = rx
            
        else:
            tx,fibre,rx     = process_data(m,tx,fibre,rx)

#%%
misc.move_files_to_folder()
misc.merge_data_folders(saving)












