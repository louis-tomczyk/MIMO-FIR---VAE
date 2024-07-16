# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.4.4
#   Date            : 2024-07-11
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
#   1.4.1 (2024-06-30) - phase noise filter parameters
#   1.4.2 (2024-07-02) - phase noise adaptative filter parameters + correction
#   1.4.3 (2024-07-10) - naming normalisation (*frame*-> *Frame*)
#   1.4.4 (2024-07-11) - file naming managed in 'create_xml_file', along with
#                           misc (1.4.0)
# ---------------------
#   2.0.0 (2024-07-12) - LIBRARY NAME CHANGED: LIB_GENERAL -> LIB_PLOT
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
import gc
import numpy            as np
import lib_misc         as misc
import lib_plot         as plot

from processing         import processing
from lib_txdsp          import set_Nsymbols
from matplotlib.pyplot  import close

from lib_matlab         import PWD
from lib_matlab         import clc
from lib_misc           import KEYS as keys
from torch              import cuda
from lib_maths          import get_power
from datetime           import date


pi = np.pi
clc()
np.set_printoptions(linewidth=160)

#%% ===========================================================================
# --- PARAMETERS ---
# =============================================================================

date                            = date.today().strftime("%Y-%m-%d")
tx,fibre,rx,saving,flags        = misc.init_dict()

###############################################################################
################################ TRANSMITTER ##################################
###############################################################################

tx["Rs"]                        = 64e9                                         # [Baud] Symbol rate
tx['SNRdB']                     = 50

tx['flag_phase_noise']          = 0
paramPHI                        = [1e5]                                        # [Hz] laser linewidth

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


# tx["mod"]                       = '64QAM'                                      # {4,16,64}QAM
# tx["nu"]                        = 0.0254                                       # for PCS: exp(-nu*|x|^2)/...

tx["mod"]                       = '16QAM'                                      # {4,16,64}QAM
tx["nu"]                        = 0.1089375                                       # for PCS: exp(-nu*|x|^2)/...

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


rx["NSymbFrame"]        = int(12800)
rx['NSymbBatch']        = int(160)



# tx['pilots_info']       = [['cpr','rand',"batchwise","polwise","4QAM",5,0]]         #ok
tx['pilots_info']       = [['cpr','rand',"same","same","4QAM",10,0]]         #ok

# tx['PhiLaw']["kind"]  = 'Rwalk'
# tx['PhiLaw']["law"]   = 'linewidth'
tx['PhiLaw']["kind"]    = 'func'
tx['PhiLaw']["law"]     = 'lin'
tx["PhiLaw"]['Start']   = 0*pi/180                                             # [rad]
tx["PhiLaw"]['End']     = 10*pi/180                                            # [rad]

win_width               = 10
tx['pn_filt_par']       = {
        'type'          : 'moving_average',
        'ma_type'       : 'gaussian',                                          # {uniform, gaussian}
        'window_size'   : win_width,
        'std_dev'       : win_width/6,                                         # only for gaussian
        'err_tolerance' : 5e-1,
        'niter_max'     : int(1e2),
        'adaptative'    : 0                                                    # {1 == yes, 0 ==  no}
}

# if tx['flag_phase_noise'] == 0:
#     tx['dnu'] = 0

###############################################################################
################################## RECEIVER ###################################
###############################################################################


rx['mode']              = 'pilots'                                             # {blind, pilots}
rx["mimo"]              = "cma"                                                # {cma,vae}

rx['SNRdB']             = 25                                                   # {>0}
# rx["lr"]                = 1e-5                                                 # {>0,<1e-2} {cma ~ 1e-5, vae ~ 5e-4}


# tauCoh                  = tx['Rs']/2/pi/tx['dnu']
# print(tauCoh)


if tx['Rs'] == 64e9:
    rx["FrameChannel"]  = 10
    rx["SymbScale"]     = 100
else:
    rx["FrameChannel"]  = 40
    rx["SymbScale"]     = 150


###############################################################################
################################## CHANNEL ####################################
###############################################################################


# if linear variations -------- [[theta_start],[theta_end],[slope]]
# if polarisation linewdith --- [[std],[NFramesChannel]]

paramPOL                        = np.array([[0],[5],[0.25]])
# paramPOL                        = np.array([[np.sqrt(2*pi*fibre['fpol']/tx['Rs'])],[10]])
# paramPOL                        = np.array([[0],[25]])

# paramLR                         = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200,300,400,500])*1e-6

if rx['mimo'].lower() == "cma":
    paramLR                         = np.array([10])*1e-6                      # {>0,<1e-2} {cma ~ 1e-5, vae é 5e-4}
else:
    paramLR                         = np.array([500])*1e-6                     # {>0,<1e-2} {cma ~ 1e-5, vae é 5e-4}

paramFIRlen                     = [9]


fibre['PMD']                    = 0.00                                         # [ps/sqrt(km)]
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

print(fibre['tauCD'])


if len(paramPOL) == 2:
    fibre['ThetasLaw']["kind"]      = 'Rwalk'
    fibre["ThetasLaw"]['law']       = 'gauss'
    fibre["ThetasLaw"]['theta_in']  = 0
else:
    fibre['ThetasLaw']["kind"]      = 'func'
    fibre['ThetasLaw']["law"]       = 'lin'





#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================
def process_data(nrea,npol,nphi,tx,fibre,rx):
        
    if len(paramPOL) == 3:      # linear evolution
        RangeTheta                  = paramPOL[1,npol] - paramPOL[0,npol]
        Sth                         = paramPOL[2,npol]
        
        deltaN_Channel              = np.floor(RangeTheta/Sth)-rx["FrameChannel"]
        NFrames_Channel             = rx["FrameChannel"]+deltaN_Channel
        Sth                         = (paramPOL[1,npol]-paramPOL[0,npol])/NFrames_Channel
        fibre["ThetasLaw"]['Start'] = (paramPOL[0,npol])*pi/180
        fibre["ThetasLaw"]['End']   = (paramPOL[1,npol])*pi/180
        
    else:
        NFrames_Channel                 = paramPOL[-1]
        fibre["ThetasLaw"]['theta_std'] = paramPOL[npol][0]
        fibre['fpol']                   = paramPOL[npol][0]

    rx["NFrames"]                       = int(rx["FrameChannel"] + NFrames_Channel)
    tx["dnu"]                           = paramPHI[nphi]                       # [Hz]
    
    tx,fibre, rx                        = set_Nsymbols(tx,fibre,rx)

    saving["filename"]                  = misc.create_xml_file(tx,fibre,rx,saving,nrea)
    tx,fibre,rx                         = processing(tx,fibre,rx,saving,flags)
        
    misc.save2mat(tx,fibre,rx,saving)

    if tx['flag_phase_noise'] == 0:
        plot.y2_axes(saving,"iteration",'Thetas','SER',['svg'])
    else:
        plot.y3_axes(saving,"iteration",'Thetas','Phis','SER',['svg'])


    # close("all")
        
    return tx,fibre,rx

#%% ===========================================================================
# --- LOGISTICS ---
# =============================================================================

Nrea = 1

for nphi in range(len(paramPHI)):
    for npol in range(len(paramPOL[0])):
        
        Nlr     = paramLR.shape[0]
        Nfir    = len(paramFIRlen)
    
        for nfir in range(Nfir):
            for nrea in range(Nrea):
                for nlr in range(Nlr):
                    
                    rx["lr"]        = paramLR[nlr]
                    tx["NSymbTaps"] = paramFIRlen[nfir]
            
                    print(f"rea={nrea}, lr={rx['lr']}, firlen={tx['NSymbTaps']}")
                    tx,fibre,rx     = process_data(nrea,npol,nphi,tx,fibre,rx)
                    
                    if "thetadiffs" in fibre:
                        del fibre['thetadiffs']
                        
                        
cuda.empty_cache()


#%%

misc.move_files_to_folder(int(date[2:4]))
misc.merge_data_folders(saving)

path = PWD(show=False)+f'/data-{date[2:]}'
misc.remove_n_characters_from_filenames(path, 20)

misc.organise_files(path)
