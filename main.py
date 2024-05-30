# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2023-03-13 : 1.0.0 - phase noise added
#                   : 2023-04-02 : 1.1.1 - phase/pol dynamics
#                   : 2023-04-19 : 1.2.0 - residual chromatic dispersion
#                                        - Fpol Law Linear
#   Date            : 2024-05-30 : 1.3.0 - cleaning, pcs table,
#   Version         : 1.3.0
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute - place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source - include copyright - state changes - include license
# 
# ----- Main idea -----
# ----- INPUTS ----- 
# ----- BIBLIOGRAPHY -----
#   Articles/Books
#   Authors             : [A0] Vincent LAUINGER
#   Title               : Blind Equalization and Channel Estimation in Coherent Optical
#                         Communications Using Variational Autoencoders
#   Jounal/Editor       : J-SAC
#   Volume - N°         : 40-9
#   Date                : 2022-11
#   DOI/ISBN            : 10.1109/JSAC.2022.3191346
#   Pages               : 2529 - 2539
#  ---------------------- 
#   Functions           : 
#   Author              : [C3] Vincent LAUINGER
#   Author contact      : vincent.lauinger@kit.edu
#   Affiliation         : Communications Engineering Lab (CEL)
#                           Karlsruhe Institute of Technology (KIT)
#   Date                : 2022-06-15
#   Title of program    : 
#   Code version        : 
#   Web Address         : https://github.com/kit-cel/vae-equalizer
# ---------------------------------------------



#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================
import lib_matlab as mb
mb.clear_all()

import numpy as np
import torch
import lib_misc as misc
from lib_misc import KEYS
import lib_txdsp as txdsp
import processing as process
import matplotlib.pyplot as plt


parallelisation = False
if parallelisation == True:
    from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda": torch.cuda.init()
torch.set_default_dtype(torch.float32)

pi = np.pi

#%% ===========================================================================
# --- PARAMETERS ---
# =============================================================================

tx,fibre,rx,saving,flags        = misc.init_dict()

# TX
tx["NSymbTaps"]                 = 7                     # must be odd number
tx["Rs"]                        = 64e9                  # Symbol rate [Baud]
tx['SNRdB']                     = 50

tx['flag_phase_noise']          = 1

# -----------------------------------------------------------------------------
# Entropies table
# -----------------------------------------------------------------------------
# 16 QAM :
#   H ------------ nu
#   4               0
#   3.75            0.1089375
#   3.5             0.16225
#   3.25            0.210875
#   3               0.2613125
#   2.75            0.3186875
#   2.49            0.3953125
#   2.25            0.50619
#   2               6.14375
# 64 QAM :
#   H ------------ nu
#   6               0
#   5.75            0.0254
#   5.5             0.038718
#   5.25            0.051203125
#   5               0.0641
#   4.75            0.078046875
#   4.5             0.0938125
#   4.25            0.112
#   4               0.133375
# -----------------------------------------------------------------------------

tx["mod"]                       = '64QAM'               # modulation format:  {4,16,64}QAM
tx["nu"]                        = .0254                # for PCS: exp(-nu*|x|^2)/sum(exp(-nu*|xi|^2))
tx['PhaseNoise_mode']           = "batch-wise"          # {samp-wise, batch-wise}

# RX
rx['SNRdB']                     = 25
rx["lr"]                        = 5e-4
rx["mimo"]                      = "cma"                 # {cma,vae}

tx['DeltaPhiC']                 = 1e-1                  # [rad]
fibre['DeltaThetaC']            = np.pi/35              # [rad]
fibre['fpol']                   = 100                   # [Hz]


# if linear variations -------- [[theta_start],[theta_end],[slope]]
# if polarisation linewdith --- [[std],[NframesChannel]]

# paramPOL                        = np.array([[5],[85],[1]])
paramPOL                        = np.array([[np.sqrt(2*pi*fibre['fpol']/tx['Rs'])],[10]])


paramPHI                        = [1e4]
paramRea                        = [1] # must be the same size as paramPOL[0]

fibre['PMD']                    = 0.04                                          # [ps/sqrt(km)]
fibre['D']                      = 17                                            # [ps/nm/km]
fibre['PMD_SI']                  = fibre['PMD']*1e-12/np.sqrt(1000)             # [s/sqrt(m)], 1e-12/np.sqrt(1000) = 1 [ps/sqrt(km)]
fibre['D_SI']                    = fibre['D']*1e-6                              # [s/m/m]
wavelength                      = 1550 *1e-9                                    # [m]
c                               = 299792458                                     # [m/s]

nSymbResCD                      = 1
beta2                           = -wavelength**2/2/pi/c*fibre['D_SI']           # [s²/rad/m]
fibre['DistRes_SI']             = nSymbResCD/(tx["Rs"]**2)/abs(beta2)/2/pi      # [m]
fibre['DistProp_SI']            = 1e4                                           # [m]
fibre['tauCD']                  = np.sqrt(2*pi*fibre['DistRes_SI']*abs(beta2))  # differential group delay [s]
# or equivalently: fibre["tauCD"] = nSymbResCD/tx['Rs']
fibre['tauPMD']                 = fibre['PMD_SI']*np.sqrt(fibre['DistProp_SI']) # PMD induced delay [s]




if len(paramPOL) == 2:
    fibre['ThetasLaw']["kind"]      = 'Rwalk'
    fibre["ThetasLaw"]['law']       = 'gauss'
    fibre["ThetasLaw"]['theta_in']  = 0
else:
    fibre['ThetasLaw']["kind"]      = 'func'
    fibre['ThetasLaw']["law"]       = 'lin'
    # fibre["ThetasLaw"]['Start']     = (9)*np.pi/180
    # fibre["ThetasLaw"]['End']       = (81)*np.pi/180


# tx['PhiLaw']["kind"]            = 'Rwalk'
# tx['PhiLaw']["law"]             = 'linewidth'
tx['PhiLaw']["kind"]            = 'func'
tx['PhiLaw']["law"]             = 'lin'
tx["PhiLaw"]['Start']           = 0*pi/180
tx["PhiLaw"]['End']             = 1*pi/180





if tx['Rs'] == 64e9:
    rx["FrameChannel"]          = 5
    rx["SymbScale"]             = 100
else:
    rx["FrameChannel"]          = 40
    rx["SymbScale"]             = 150


#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================
def process_data(npol,nphi,tx,fibre,rx):

    print(paramPOL[:,npol])

    if len(paramPOL) == 3:      # linear evolution
        RangeTheta                      = paramPOL[1,npol] - paramPOL[0,npol]
        SlopeTheta                      = paramPOL[2,npol]
        
        deltaN_Channel                  = np.floor(RangeTheta/SlopeTheta)-rx["FrameChannel"]
        Nframes_Channel                 = rx["FrameChannel"]+deltaN_Channel
        slope                           = (paramPOL[1,npol]-paramPOL[0,npol])/Nframes_Channel
        print(slope)
        fibre["ThetasLaw"]['Start']     = (paramPOL[0,npol])*np.pi/180
        fibre["ThetasLaw"]['End']       = (paramPOL[1,npol])*np.pi/180
    else:
        Nframes_Channel                 = paramPOL[-1]
        fibre["ThetasLaw"]['theta_std'] = paramPOL[npol][0]

    rx["Nframes"]                       = int(rx["FrameChannel"] + Nframes_Channel)


    tx["dnu"]                           = paramPHI[nphi]    # [Hz]
    tx,fibre, rx                        = txdsp.set_Nsymbols(tx,fibre,rx)

    saving["filename"]                  = misc.create_xml_file(tx,fibre,rx,saving)[2:-4]
    tx,fibre,rx                         = process.processing(tx,fibre,rx,saving,flags)
        
    misc.save2mat(tx,fibre,rx,saving)

    if rx['mimo'].lower() == "cma":
        misc.plot_2y_axes(saving,"iteration",'Thetas','loss',['svg'])
    else:
        misc.plot_3y_axes(saving,"iteration",'SNR','Thetas','loss',['svg'])
    

    if Nrea > 1:
        plt.close("all")
        
    return tx,fibre,rx

#%% ===========================================================================
# --- LOGISTICS ---
# =============================================================================

for npol in range(len(paramPOL[0])):

    
    Nrea = int(paramRea[npol])
    print(Nrea)

    for k in range(Nrea):

        for nphi in range(len(paramPHI)):
            if parallelisation == True:
                results     = Parallel(n_jobs=2)(
                    delayed(process_data)(k,tx,fibre,rx) for k in range(len(paramPOL)))
                
            else:
                tx,fibre,rx     = process_data(npol,nphi,tx,fibre,rx)
                
                if "thetadiffs" in fibre:
                    del fibre['thetadiffs']


#%%

misc.move_files_to_folder()
misc.merge_data_folders(saving)
