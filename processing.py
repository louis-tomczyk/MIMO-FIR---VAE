# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.3.0
#   Date            : 2024-07-01
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
#   1.1.0 (2024-03-27) - print_result - PhaseNoise management
#                      - [NEW] save estimations
#   1.1.1 (2024-04-03) - print_result
#   1.2.0 (2024-04-19) - init_processing, along with rxdsp (1.1.0)
#   1.2.1 (2024-05-22) - init_processing, along with rxdsp (1.3.0)
#                      - init_processing, print_result - differentiating
#                           VAE-CMA for initialisation
#   1.2.2 (2024-05-28) - print_result
#   1.2.3 (2024-06-04) - init_processing, pilots management, along with txdsp
#                           (1.1.0)
#   1.2.4 (2024-06-07) - init_processing NSymbConv and NSymbEq changed to
#                           symbol wise instead of weird mix, along with rxdsp
#                           (1.4.1)
#                      - cleaning (Ntaps, Nconv)
#   1.2.5 (2024-06-20) - init_processing: pilots for CPR, see rxdsp.CPR_pilots
#                           along with rxdsp (1.5.0)
#   1.2.6 (2024-06-21) - init_processing: avoid 'pilots' with "vae"
#   1.3.0 (2024-07-01) - init_processing: applying same learning scheme for cma
#                      - print_results: merging cma/vae display of results
#   1.3.1 (2024-07-02) - init_processing: Frame-FrameChannel for phase noise =>
#                                           Frame, along with rxdsp (1.6.1)
# 
# ----- MAIN IDEA -----
#   Simulation of an end-to-end linear optical telecommunication system
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books
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
#        Web Address     : https://github.com/kit-cel/vae-equalizer
# ---------------------------------------------
# %%


#%% ===========================================================================
# --- CONTENTS ---
# =============================================================================
# * processing_self
# - init_processing
# - init_train
# - print_results
# - save_data
# - save_estimations
# =============================================================================


#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time

import lib_kit as kit
import lib_misc as misc
import lib_prop as prop
import lib_txdsp as txdsp
import lib_txhw as txhw
import lib_rxdsp as rxdsp
import lib_general as gen
import lib_matlab as mat
from lib_matlab import clc
import timeit

from lib_matlab import clc
from lib_misc import KEYS as keys
from lib_maths import get_power as power
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================

#%%
def processing(tx, fibre, rx, saving, flags):

    tx, fibre, rx       = init_processing(tx, fibre, rx, saving, device)

    for frame in range(rx['Nframes']):

        rx              = init_train(tx, rx, frame)                            # if vae, otherwise: transparent
        tx              = txdsp.transmitter(tx, rx)
        tx, fibre, rx   = prop.propagation(tx, fibre, rx)
        rx, loss        = rxdsp.receiver(tx,rx,saving)
        array           = print_results(loss, frame, tx, fibre, rx, saving)

    tx, fibre, rx = save_data(tx, fibre, rx, saving, array)

    return tx, fibre, rx



#%%

def init_processing(tx, fibre, rx, saving, device):
# =============================================================================
# MISSING FIELDS
# =============================================================================

    flag_incoherence = 0
    if rx['mimo'].lower() == 'vae' and rx['mode'] != 'blind':
        flag_incoherence = 1
        
    assert not flag_incoherence, "pilots aided equalisation is for CMA only\
                                check rx['mimo'] in main.py"
    
    if "channel" not in fibre:
        fibre['channel']    = "linear"
    
    if 'shaping_filter' not in tx and fibre["channel"].lower() != "awgn":
        tx['shaping_filter']= "rrc"

    if 'skip' not in saving:
        saving['skip']      = 10
        
    if 'PhaseNoise_mode' not in tx:
        tx['PhaseNoise_mode'] = "batch-wise"                                   # {samp-wise, batch-wise}
        
        
    if 'DeltaPhiC' not in tx:
        tx['DeltaPhiC']                 = 1e-1                                 # [rad]
        
    if 'DeltaThetaC' not in fibre:
        fibre['DeltaThetaC']            = np.pi/35                             # [rad]

# =============================================================================
# SYMBOLS AND SAMPLES MANAGEMENT
# =============================================================================

    rx['NBatchFrame']       = int(rx['NSymbFrame']/rx['NSymbBatch'])
    rx['NsampFrame']        = rx["NsampBatch"]*rx['NBatchFrame']

    rx["NframesChannel"]    = rx["Nframes"]-rx["FrameChannel"]   
    rx["NframesTraining"]   = rx["Nframes"]-rx["NframesChannel"]
     
    rx["NBatchesChannel"]   = rx["NframesChannel"]*rx['NBatchFrame'] 
    rx["NBatchesTot"]       = rx["Nframes"]*rx['NBatchFrame']
    rx["NBatchesTraining"]  = rx["NBatchesTot"]-rx['NBatchesChannel']

    
    tx['NsampChannel']      = tx['NsampFrame']*rx['NframesChannel']
    tx['Nsamptraining']     = tx['NsampTot']-tx['NsampChannel'] 

# =============================================================================
# TRANSMITTER
# =============================================================================

    tx['Npolars']           = 2
    tx["RollOff"]           = 0.1
    tx["fs"]                = (1+tx["RollOff"])/2*tx["Rs"]*tx["Nsps"]           # [GHz]
    tx, rx                  = txdsp.get_constellation(tx, rx)
    
    if rx['mode'].lower()   != "blind":
        for k in range(len(tx['pilots_info'])):
            tx, rx           = txdsp.get_constellation(tx, rx, tx['pilots_info'][k])

        
    tx["PhaseNoise"]        = np.zeros((tx['Npolars'],tx['NsampFrame'], rx['Nframes']))
    tx['mimo']              = rx['mimo'] # usefull for txdsp.pilot_generation
    tx['NSymbBatch']        = rx['NSymbBatch']
    tx['NsampBatch']        = rx['NsampBatch']

# =============================================================================
# INITIALISATION OF CHANNEL MATRIX
# =============================================================================


    h_est = np.zeros([tx['Npolars'], tx['Npolars'], 2, tx['NsampTaps']])
    h_est[0,0,0,tx["NSymbTaps"]-1] = 1
    h_est[1,1,0,tx["NSymbTaps"]-1] = 1

    if rx['mimo'].lower() == "vae":
        h_est               = misc.my_tensor(h_est, requires_grad=True)

    rx["h_est"]             = h_est
    fibre['phiIQ']          = np.zeros(tx['Npolars'],dtype = complex)

# =============================================================================
# RANDOM EFFECTS
# =============================================================================

    fibre                   = prop.set_thetas(tx, fibre, rx)
    tx                      = txhw.gen_phase_noise(tx, rx)

# =============================================================================
# RECEIVER
# =============================================================================

    if rx['mimo'].lower() == "vae":
        rx['net']           = kit.twoXtwoFIR(tx).to(device)
        rx['optimiser']     = optim.Adam(rx['net'].parameters(), lr=rx['lr'])
        rx['optimiser'].add_param_group({"params": rx["h_est"]})

    if rx['mode'].lower() != 'blind':
        # we remove the first and last batch of each frame for edges effects
        rx['NBatchFrame_pilots']= rx['NBatchFrame']-2
        rx['NSymb_pilots_cpr']  = tx['NSymb_pilots_cpr']-tx['NSymbTaps']
        rx['Nzeros_stuffing']   = rx['NSymbBatch']-rx['NSymb_pilots_cpr']
        rx['PhaseNoise_pilots'] = np.zeros((rx['Nframes'],tx['Npolars'],\
                                            rx['NBatchFrame_pilots']))
        rx['PhaseNoise_pilots_std'] = np.zeros(rx['PhaseNoise_pilots'].shape)
            
        rx['NSymbEq']           = rx["NSymbFrame"]-2*rx['NSymbBatch']


# =============================================================================
# OUTPUT
# =============================================================================

    rx["H_est_l"]           = []
    rx["H_lins"]            = []
    rx["Losses"]            = []
    rx["SNRdBs"]            = []
    rx["SERs"]              = []


    rx["SNRdB_est"]         = np.zeros(rx['Nframes'])
    rx['sig_real']          = torch.zeros((tx['Npolars']*2,tx['NsampFrame']))



    if rx['mimo'].lower() == "vae":
        rx['sig_eq_real']   = np.zeros((tx['Npolars']*2,
                                    rx['Nframes'],
                                    rx['NBatchFrame'],
                                    rx['NSymbBatch'])).astype(np.float32)
    
    else:
        rx['sig_eq_real']   = np.zeros((tx['Npolars']*2,
                                    rx['Nframes'],
                                    rx['NSymbEq'])).astype(np.float32)



    rx['Symb_real_dec']     = np.zeros((tx['Npolars']*2,
                                        rx['Nframes'],
                                        rx['NSymbEq']),dtype = np.float16)
    
    rx["SER_valid"]         = np.zeros((2, rx['Nframes']),dtype = np.float128)
    rx['Pnoise_est']        = np.zeros((tx["Npolars"],
                                        rx['Nframes']),dtype = np.float32)


    tx      = misc.sort_dict_by_keys(tx)
    fibre   = misc.sort_dict_by_keys(fibre)
    rx      = misc.sort_dict_by_keys(rx)

    return tx, fibre, rx



# %%
def init_train(tx, rx, frame):

    # if rx["N_lrhalf"] > rx["Nframes"]: # [C1]

    #     rx["lr_scheduled"] = rx["lr"] * 0.5
    #     rx['optimiser'].param_groups[0]['lr'] = rx["lr_scheduled"]


    rx["Frame"]             = frame

    if rx["mimo"].lower() == 'vae':
        with torch.set_grad_enabled(True):
            rx['out_train']             = misc.my_zeros_tensor((tx["Npolars"], 2*tx["N_amps"], rx["NSymbFrame"]))
            rx['Pnoise_batches']        = misc.my_zeros_tensor((tx["Npolars"], rx['NBatchFrame']))
            rx['out_const']             = misc.my_zeros_tensor((tx["Npolars"], 2, rx["NSymbFrame"]))
            rx['losses_subframe']       = misc.my_zeros_tensor((rx["Nframes"], rx['NBatchFrame']))
            rx['DKL_subframe']          = misc.my_zeros_tensor((rx["Nframes"], rx['NBatchFrame']))
            rx['Llikelihood_subframe']  = misc.my_zeros_tensor((rx["Nframes"], rx['NBatchFrame']))

            rx['net'].train()
            rx['optimiser'].zero_grad()

    rx = misc.sort_dict_by_keys(rx)

    return rx




# %%
def print_results(loss, frame, tx, fibre, rx, saving):

    SER_valid   = rx["SER_valid"]
    SERs        = rx["SERs"]
    Losses      = rx["Losses"]

    if rx["mimo"].lower() == "vae":
        SNRdB_est       = rx["SNRdB_est"][frame].item()
        SNRdBs          = rx["SNRdBs"]
        lossk           = round(loss.item(), 0)
        SNRdBk          = round(10*np.log10(SNRdB_est), 5)
        SNRdBs.append(SNRdBk)
        
        SNRdBs          = misc.list2vector(SNRdBs)
        SERvalid0       = round(SER_valid[0, frame].item(), 15)
        SERvalid1       = round(SER_valid[1, frame].item(), 15)

        SERsvalid       = np.array([SERvalid0, SERvalid1])
        SERmeank        = round(np.mean(SERsvalid), 15)

    else:
        
        if rx['mimo'].lower() == "vae":
            lossk   = torch.std(loss[-1][:]).item()
        else:
            lossk   = np.std(loss[-1][:]).item()

        SERvalidH   = round(SER_valid[0, rx['Frame']].item(), 15)
        SERvalidV   = round(SER_valid[1, rx['Frame']].item(), 15)

        SERsvalid   = np.array([SERvalidH, SERvalidV])
        SERmeank    = round(np.mean(SERsvalid), 15)


    SERs.append(SERmeank)
    Losses.append(lossk)

    Losses      = misc.list2vector(Losses)
    SERs        = misc.list2vector(SERs)
    Iteration   = mat.linspace(1, len(Losses), len(Losses))
    
    if rx['mimo'].lower() == "vae":
        array   = np.concatenate((Iteration, Losses, SNRdBs, SERs), axis=1)
    else:
        array   = np.concatenate((Iteration,Losses,SERs),axis=1)
        # array = np.concatenate((Iteration, Losses), axis=1)

    if rx['Frame'] >= rx["FrameChannel"]:
        thetak = fibre['thetas'][rx['Frame']][1]+fibre['thetas'][rx['Frame']][0]
    else:
        thetak = 0

    #'''
    if rx["mimo"].lower() == 'vae':
        if tx['flag_phase_noise'] == 1:
            print("frame %d" % frame,
                  '--- loss     = %.1f'     % lossk,
                  '--- SNRdB    = %.2f'     % SNRdBk,
                  '--- Theta    = %.2f'     % (thetak*180/np.pi),
                  '--- std(Phi) = %.1f'     % (np.std(tx["PhaseNoise"][0, :,rx['Frame']])*180/np.pi),
                  '--- <SER>    = %.2e'     % SERmeank,
                  )
        else:
            print("frame %d" % frame,
                  '--- loss     = %.1f'     % lossk,
                  '--- SNRdB    = %.2f'     % SNRdBk,
                  '--- Theta    = %.2f'     % (thetak*180/np.pi),
                  '--- <SER>    = %.2e'     % SERmeank,
                  )
            
    else:
        if tx['flag_phase_noise'] == 1:
            print("frame %d" % frame,
                  '--- loss     = %.3e'     % lossk,
                  '--- Theta    = %.2f'     % (thetak*180/np.pi),
                  '--- std(Phi) = %.1f'     % (np.std(tx["PhaseNoise"][0, :, rx["Frame"]])*180/np.pi),
                  '--- <SER>    = %.2e'     % SERmeank,
                  )
        else:
            print("frame %d" % frame,
                  '--- loss     = %.3e'     % lossk,
                  '--- Theta    = %.2f'     % (thetak*180/np.pi),
                  '--- <SER>    = %.2e'     % SERmeank,
                  )
    #'''

    return array



#%%
def save_data(tx, fibre, rx, saving, array):

    Thetas_IN   = np.array([fibre["thetas"][k][0] for k in range(rx["Nframes"])])
    Thetas_OUT  = np.array([fibre["thetas"][k][1] for k in range(rx["Nframes"])])
    thetas      = list((Thetas_IN+Thetas_OUT)*180/np.pi)
    Thetas      = misc.list2vector(thetas)

    # PhisAll         = tx['PhaseNoise'][0, :, :]
    # PhiMeanFrame    = np.mean(PhisAll,axis=0)
    # PhiStdFrame     = np.expand_dims(np.std(PhisAll, axis=0), axis=1)
    # array2          = np.concatenate((array, Thetas, PhiStdFrame), axis=1)
    array2          = np.concatenate((array, Thetas), axis=1)


    if rx["mimo"].lower() == "vae":
        # misc.array2csv(array2, saving["filename"], ["iteration",
        #                 "loss", "SNR", "SER", "Thetas", 'std(Phi)'])
        misc.array2csv(array2, saving["filename"], ["iteration",
                       "loss", "SNR", "SER", "Thetas"])
    else:
        misc.array2csv(array2, saving["filename"], ["iteration", "loss", "SER", "Thetas"])
        # misc.array2csv(array2, saving["filename"], ["iteration", "loss", "SER"])

    tx      = misc.sort_dict_by_keys(tx)
    fibre   = misc.sort_dict_by_keys(fibre)
    rx      = misc.sort_dict_by_keys(rx)

    return tx, fibre, rx



