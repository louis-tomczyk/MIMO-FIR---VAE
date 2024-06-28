# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.4.0
#   Date            : 2024-06-27
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2023-03-13) - phase noise added
#   1.1.1 (2023-04-02) - phase/pol dynamics
#   1.2.0 (2023-04-19) - residual chromatic dispersion, Fpol Law Linear
#   1.3.0 (2024-05-27) - cleaning, pcs table
#   1.3.1 (2024-06-04) - blind/pilot aided
#   1.3.2 (2024-06-07) - [DRAFT] blind/pilot aided, cleaning
#   1.3.2 (2024-06-14) - cazac
#   1.4.0 (2024-06-27) - folder management: no more temp folders
#                        instead create_xml_file adds the realisation number
#
# ----- MAIN IDEA -----
#   Simulation of an end-to-end linear optical telecommunication system
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   [A1] Authors         : Vincent Lauinger
#        Title           : Blind Equalization and Channel Estimation in
#                          Coherent Optical Communications Using Variational
#                          Autoencoders
#        Journal/Editor  : J-SAC
#        Volume - N°     : 40-9
#        Date            : 2022-11
#        DOI/ISBN        : 10.1109/JSAC.2022.3191346
#        Pages           : 2529 - 2539
#  ----------------------
#   CODE
#   [C1] Author          : Vincent Lauinger
#        Contact         : vincent.lauinger@kit.edu
#        Laboratory/team : Communications Engineering Lab
#        Institution     : Karlsruhe Institute of Technology (KIT)
#        Date            : 2022-06-15
#        Program Title   : 
#        Code Version    : 
#        Type            : Source code
#        Web Address     : https://github.com/kit-cel/vae-equalizer
# ---------------------------------------------


#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================
import lib_matlab as mb
mb.clear_all()

import numpy as np
import torch
import lib_misc as misc
import lib_txdsp as txdsp
import processing as process
import matplotlib.pyplot as plt


parallelisation = False
if parallelisation == True:
    from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda": torch.cuda.init()
torch.set_default_dtype(torch.float32)


from lib_matlab import clc
from lib_misc import KEYS as keys
from lib_maths import get_power as power
import lib_matlab as mb

pi = np.pi
clc()
np.set_printoptions(linewidth=160)

#%% ===========================================================================
# --- PARAMETERS ---
# =============================================================================

tx,fibre,rx,saving,flags        = misc.init_dict()

date = '2024-06-28'
###############################################################################
################################ TRANSMITTER ##################################
###############################################################################

tx["NSymbTaps"]                 = 7                                           # must be odd number
tx["Rs"]                        = 64e9                                         # Baud] Symbol rate
tx['SNRdB']                     = 90

tx['flag_phase_noise']          = 0

paramPHI                        = [1e6]                                        # [Hz] laser linewidth

# -----------------------------------------------------------------------------
# Entropies table
# -----------------------------------------------------------------------------


# -------------------------------------------------------
# |        16 QAM :          |        64 QAM :          |
# |--------------------------|--------------------------|
# | H        | nu            | H        | nu            |
# |----------|---------------|----------|---------------|
# | 4        | 0             | 6        | 0             |
# | 3.75     | 0.1089375     | 5.75     | 0.0254        |
# | 3.5      | 0.16225       | 5.5      | 0.038718      |
# | 3.25     | 0.210875      | 5.25     | 0.051203125   |
# | 3        | 0.2613125     | 5        | 0.0641        |
# | 2.75     | 0.3186875     | 4.75     | 0.078046875   |
# | 2.49     | 0.3953125     | 4.5      | 0.0938125     |
# | 2.25     | 0.50619       | 4.25     | 0.11          |
# | 2        | 6.14375       | 4        | 0.133375      |
# -------------------------------------------------------


tx["mod"]                       = '16QAM'                                      # {4,16,64}QAM
tx["nu"]                        = 0#0.0254                                       # for PCS: exp(-nu*|x|^2)/...


# -----------------------------------------------------------------------------
# Pilots management
# -----------------------------------------------------------------------------
# parameters description
# 0 = {cpr, synchro_once, synchro_frame} ======== pilots locations
# 1 = {rand, file, custom, cazac, data, ...} ==== pilots selection method
# 2 = {same, batchwise, framewise} ============== pilots changes?
#                                                   batchwise : cpr
#                                                   framewise : synchro(_once)
# 3 = {same, polwise} =========================== same for both polarisations?
# 4 = {4, 16, 64}QAM ============================ modulation format used
# 5 = {>0} ====================================== percentage of pilots
#                                                   if not cazac
#                                                 number of cazac symbol
#                                                   otherwise
# 6 = {>0} ====================================== number of pilots per batch
#                                                   if not cazac
# -----------------------------------------------------------------------------
# Examples
# tx['pilots_info']       = [['synchro','rand',"framewise","same","4QAM",3,0]]   #ok
# tx['pilots_info']       = [['synchro','rand',"framewise","polwise","4QAM",3,0]]#ok
# tx['pilots_info']       = [['synchro','cazac',"framewise","","",64]]           #ok
# tx['pilots_info']       = [['synchro','data',"framewise","same","",10]]        #ok
# tx['pilots_info']       = [['synchro','data',"framewise","polwise","",10]]     #ok

# tx['pilots_info']       = [['synchro_once','rand',"","same","4QAM",3,0]]       #ok
# tx['pilots_info']       = [['synchro_once','rand',"","polwise","4QAM",3,0]]    #ok
# tx['pilots_info']       = [['synchro_once','cazac',"","","",64]]               #ok
# tx['pilots_info']       = [['synchro_once','data',"","polwise","",10]]         #ok
# tx['pilots_info']       = [['synchro_once','data',"","same","",10]]            #ok

# tx['pilots_info']       = [['cpr','rand',"same","same","4QAM",3,0]]            #ok
# tx['pilots_info']       = [['cpr','rand',"same","polwise","4QAM",3,0]]         #ok
# tx['pilots_info']       = [['cpr','rand',"batchwise","same","4QAM",3,0]]       #ok
# tx['pilots_info']       = [['cpr','rand',"batchwise","polwise","4QAM",3,0]]    #ok
# tx['pilots_info']       = [['cpr','cazac',"","","",64]]                        #ok
# -----------------------------------------------------------------------------


# tx['pilots_info']       = [['synchro_once','data',"","same","",10],
#                            ['cpr','rand',"same","same","4QAM",3,0]]



# tx['pilots_info']       = [['cpr','rand',"batchwise","polwise","4QAM",5,0]]         #ok
tx['pilots_info']       = [['cpr','rand',"same","same","4QAM",10,0]]         #ok

# tx['PhiLaw']["kind"]  = 'Rwalk'
# tx['PhiLaw']["law"]   = 'linewidth'
tx['PhiLaw']["kind"]    = 'func'
tx['PhiLaw']["law"]     = 'lin'
tx["PhiLaw"]['Start']   = 10*pi/180                                             # [rad]
tx["PhiLaw"]['End']     = 25*pi/180                                            # [rad]


###############################################################################
################################## RECEIVER ###################################
###############################################################################

rx['mode']              = 'blind'                                             # {blind, pilots}
rx["mimo"]              = "cma"                                                # {cma,vae}

rx['SNRdB']             = 25                                                   # {>0}
rx["lr"]                = 1e-5                                                 # {>0,<1e-2} {cma ~ 1e-5, vae é 5e-4}

if tx['Rs'] == 64e9:
    rx["FrameChannel"]  = 15
    rx["SymbScale"]     = 100
else:
    rx["FrameChannel"]  = 40
    rx["SymbScale"]     = 150


###############################################################################
################################## CHANNEL ####################################
###############################################################################


# if linear variations -------- [[theta_start],[theta_end],[slope]]
# if polarisation linewdith --- [[std],[NframesChannel]]

paramPOL                        = np.array([[0],[10],[3]])
# paramPOL                        = np.array([[np.sqrt(2*pi*fibre['fpol']/tx['Rs'])],[10]])
# paramPOL                        = np.array([[0],[10]])


paramRea                        = [1] # must be the same size as paramPOL[0]

fibre['PMD']                    = 0.0                                         # [ps/sqrt(km)]
fibre['D']                      = 17                                           # [ps/nm/km]
fibre['PMD_SI']                 = fibre['PMD']*1e-12/np.sqrt(1000)             # [s/sqrt(m)], 1e-12/np.sqrt(1000) = 1 [ps/sqrt(km)]
fibre['D_SI']                   = fibre['D']*1e-6                              # [s/m/m]
wavelength                      = 1550 *1e-9                                   # [m]
c                               = 299792458                                    # [m/s]

nSymbResCD                      = 0
beta2                           = -wavelength**2/2/pi/c*fibre['D_SI']          # [s²/rad/m]
fibre['DistRes_SI']             = nSymbResCD/(tx["Rs"]**2)/abs(beta2)/2/pi     # [m]
fibre['DistProp_SI']            = 1e4                                          # [m]









fibre['tauCD']                  = np.sqrt(2*pi*fibre['DistRes_SI']*abs(beta2)) # differential group delay [s]
fibre['tauPMD']                 = fibre['PMD_SI']*np.sqrt(fibre['DistProp_SI'])# PMD induced delay [s]




if len(paramPOL) == 2:
    fibre['ThetasLaw']["kind"]      = 'Rwalk'
    fibre["ThetasLaw"]['law']       = 'gauss'
    fibre["ThetasLaw"]['theta_in']  = 0
else:
    fibre['ThetasLaw']["kind"]      = 'func'
    fibre['ThetasLaw']["law"]       = 'lin'
    # fibre["ThetasLaw"]['Start']     = (9)*np.pi/180
    # fibre["ThetasLaw"]['End']       = (81)*np.pi/180


rx["NSymbFrame"]    = 20000
rx['NSymbBatch']    = 1000


#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================
def process_data(nrea,npol,nphi,tx,fibre,rx):

    path = mb.PWD(show=False)
    for k in range(len(paramPOL)):
        print("{}".format(paramPOL[k,npol]))
        
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
        fibre['fpol']                   = paramPOL[npol][0]

    rx["Nframes"]                       = int(rx["FrameChannel"] + Nframes_Channel)


    tx["dnu"]                           = paramPHI[nphi]                       # [Hz]
    
    tx,fibre, rx                        = txdsp.set_Nsymbols(tx,fibre,rx)

    saving["filename"]                  = misc.create_xml_file(tx,fibre,rx,saving,nrea)[2:-4]
    tx,fibre,rx                         = process.processing(tx,fibre,rx,saving,flags)
        
    misc.save2mat(tx,fibre,rx,saving)

    if rx['mimo'].lower() == "cma":
        misc.plot_2y_axes(saving,"iteration",'Thetas','SER',['svg'])
    else:
        misc.plot_3y_axes(saving,"iteration",'SNR','Thetas','SER',['svg'])
    

    if Nrea > 1:
        plt.close("all")
    
    misc.replace_string_in_filenames(path, "NSymbBatch","NSbB")
    misc.replace_string_in_filenames(path, "NSymbFrame","NSbF")
    misc.replace_string_in_filenames(path, "NsampTaps","NspT")
    misc.truncate_lr_in_filename(path)
    
        
    return tx,fibre,rx

#%% ===========================================================================
# --- LOGISTICS ---
# =============================================================================

for npol in range(len(paramPOL[0])):

    Nrea = 2

    for nrea in range(Nrea):

        for nphi in range(len(paramPHI)):
            if parallelisation == True:
                results     = Parallel(n_jobs=2)(
                    delayed(process_data)(nrea,k,tx,fibre,rx) for k in range(len(paramPOL)))
                
            else:
                tx,fibre,rx     = process_data(nrea,npol,nphi,tx,fibre,rx)
                
                if "thetadiffs" in fibre:
                    del fibre['thetadiffs']


#%%

misc.move_files_to_folder(int(date[2:4]))
misc.merge_data_folders(saving)

path = mb.PWD(show=False)+f'/data-{date[2:]}'
misc.remove_n_characters_from_filenames(path, 20)


