# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.3.12
#   Date            : 2024-11-05
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
#                                         Frame, along with rxdsp (1.6.1)
#   1.3.2 (2024-07-05) - init_processing: imported from rxdsp.compensate_and_
#                           truncate definition of t/rx['Symb_SER_real']
#                           as I was erasing decided symbols at each frame b4.
#                           along with rxdsp (1.6.2)
#                        cleaning names: sig_eq = sig_mimo_cut + sig_cpr while
#                           same name for multiple different vectors. along
#                           with kit (1.1.4)
#   1.3.3 (2024-07-07) - init_processing: front end normalisation according to
#                           mimo algorithm
#   1.3.4 (2024-07-10) - naming normalisation (*frame*-> *Frame*).
#                        along with main (1.4.3)
#                      - init_processing: saving per frame/bacth?
#   1.3.5 (2024-07-11) - save_data: managing phase noise
#   1.3.6 (2024-07-16) - init_processing: managing offset for pilots removal
#                           previously manually adjusted in rxdsp
#   1.3.7 (2024-07-17) - init_processing, print_results, save_data: managing
#                           elapsed time
#   1.3.8 (2024-07-18) - save_data: array columns reodered correctly after
#                           adding "dt"
#   1.3.9 (2024-07-24) - print_results: server management
#   1.3.10(2024-10-03) - seed management for random generation, along with main
#                           (2.0.4)
#   1.3.11(2024-10-07) - init_processing SYMBOLS AND SAMPLES MANAGEMENT moved
#                           to txdsp.set_Nsymbols, txdsp (2.0.2)
#   1.3.12(2024-11-05) - print_result: tx['get_time'] instead of 'get_exec_time
#                         tx['flag_phase_noise] -> tx['flag_phase]
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
import random
import time

import lib_kit as kit
import lib_misc as misc
import lib_prop as prop
import lib_txdsp as txdsp
import lib_txhw as txhw
import lib_rxdsp as rxdsp
import lib_plot as plot
import lib_matlab as mat
from lib_matlab import clc


from lib_matlab import clc
from lib_misc import KEYS as keys
from lib_maths import get_power as power
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
pi      = np.pi

#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================

#%%
def processing(tx, fibre, rx, saving,seed_id):

    tx, fibre, rx       = init_processing(tx, fibre, rx, saving, device)

    for frame in range(rx['NFrames']):

        seed_id = seed_id+frame
        random.seed(seed_id)
        
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

    if 'save_channel_gnd' not in rx:
        rx['save_channel_gnd'] = 0
        
    rx['save_channel_frame'] = 1
    
    if tx['flag_phase']:
        rx['save_channel_batch'] = 1
        
    else:
        rx['save_channel_batch'] = 1

        
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

    # rx['NBatchFrame']       = int(rx['NSymbFrame']/rx['NSymbBatch'])
    # rx['NsampFrame']        = rx["NsampBatch"]*rx['NBatchFrame']

    # rx["NFramesChannel"]    = rx["NFrames"]-rx["FrameChannel"]   
    # rx["NFramesTraining"]   = rx["NFrames"]-rx["NFramesChannel"]
     
    # rx["NBatchesChannel"]   = rx["NFramesChannel"]*rx['NBatchFrame'] 
    # rx["NBatchesTot"]       = rx["NFrames"]*rx['NBatchFrame']
    # rx["NBatchesTraining"]  = rx["NBatchesTot"]-rx['NBatchesChannel']

    # tx['NsampChannel']      = tx['NsampFrame']*rx['NFramesChannel']
    # tx['Nsamptraining']     = tx['NsampTot']-tx['NsampChannel'] 

    
        
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

        
    tx["PhaseNoise"]        = np.zeros((tx['Npolars'],tx['NsampFrame'], rx['NFrames']))
    tx['mimo']              = rx['mimo'] # usefull for txdsp.pilot_generation
    tx['NSymbBatch']        = rx['NSymbBatch']
    tx['NsampBatch']        = rx['NsampBatch']
    
    # useless condition for now
    if rx['mimo'].lower() == "cma":
        tx['norm_power'] = 0
        rx['norm_power'] = 1

    else:
        tx['norm_power'] = 0
        rx['norm_power'] = 0


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
# CHANNEL EFFECTS
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
        rx['Nzeros_stuffing']   = rx['NSymbBatch']-rx['NSymb_pilots_cpr']
        rx['PhaseNoise_pilots'] = np.zeros((rx['NFrames'],tx['Npolars'],\
                                            rx['NBatchFrame_pilots']))
        rx['PhaseNoise_pilots_std'] = np.zeros(rx['PhaseNoise_pilots'].shape)
            
        rx['NSymbEq']   = rx["NSymbFrame"]-2*rx['NSymbBatch']        
        rx['NSymbSER']  = rx['NSymbEq'] - rx['NBatchFrame_pilots']*(rx['NSymb_pilots_cpr'])
        
        # shaping using convolution puts symboles at end of previous batch
        # and at beginning of current batch. the number of the latter is named
        # ``offset_conv"
        
        rx['offset_conv_begin'] = rx['NSymb_pilots_cpr'] - tx['NSymbTaps']+1
        rx['offset_conv_end'] = rx['NSymb_pilots_cpr'] - rx['offset_conv_begin']

    else:
        if rx['mimo'].lower() == "vae":
            rx['NSymbSER'] = rx["NSymbFrame"]
        else:
            rx['NSymbSER'] = rx["NSymbFrame"]-rx['NSymbCut_tot']+1
            
        

            
# =============================================================================
# OUTPUT
# =============================================================================

    rx["h_est_frame"]       = []

    if rx['save_channel_batch']:
        rx["h_est_batch"]   = []

    if rx['save_channel_gnd']:
        rx["h_gnd"]         = []
        
    rx["Losses"]            = []
    rx["SNRdBs"]            = []
    rx["SERs"]              = []


    rx["SNRdB_est"]         = np.zeros(rx['NFrames'])
    rx['sig_real']          = torch.zeros((tx['Npolars']*2,tx['NsampFrame']))



    if rx['mimo'].lower() == "vae":
        rx["sig_mimo_real"]   = np.zeros((tx['Npolars']*2,
                                    rx['NFrames'],
                                    rx['NBatchFrame'],
                                    rx['NSymbBatch'])).astype(np.float32)
    
    
    rx['Symb_real_dec']     = np.zeros((tx['Npolars']*2,
                                        rx['NFrames'],
                                        rx['NSymbSER']),dtype = np.float16)
            
            
    tx['Symb_SER_real'] = np.zeros((tx['Npolars']*2,
                                    rx['NFrames'],
                                    rx['NSymbSER'])).astype(np.float16)
    
    rx['Symb_SER_real'] = np.zeros((tx['Npolars']*2,
                                    rx['NFrames'],
                                    rx['NSymbSER'])).astype(np.float16)
    
    rx["SER_valid"]         = np.zeros((2, rx['NFrames']),dtype = np.float128)
    rx['Pnoise_est']        = np.zeros((tx["Npolars"],
                                        rx['NFrames']),dtype = np.float32)


    if 'get_exec_time' in tx:
        tx['get_exec_time'].append(np.zeros((1,rx['NFrames'])))
        tx['get_exec_time'][1] = time.time()
        
        
    tx      = misc.sort_dict_by_keys(tx)
    fibre   = misc.sort_dict_by_keys(fibre)
    rx      = misc.sort_dict_by_keys(rx)

    return tx, fibre, rx



# %%
def init_train(tx, rx, frame):

    # if rx["N_lrhalf"] > rx["NFrames"]: # [C1]

    #     rx["lr_scheduled"] = rx["lr"] * 0.5
    #     rx['optimiser'].param_groups[0]['lr'] = rx["lr_scheduled"]


    rx["Frame"]             = frame

    if rx["mimo"].lower() == 'vae':
        with torch.set_grad_enabled(True):
            rx['out_train']             = misc.my_zeros_tensor((tx["Npolars"], 2*tx["N_amps"], rx["NSymbFrame"]))
            rx['Pnoise_batches']        = misc.my_zeros_tensor((tx["Npolars"], rx['NBatchFrame']))
            rx['out_const']             = misc.my_zeros_tensor((tx["Npolars"], 2, rx["NSymbFrame"]))
            rx['losses_subframe']       = misc.my_zeros_tensor((rx["NFrames"], rx['NBatchFrame']))
            rx['DKL_subframe']          = misc.my_zeros_tensor((rx["NFrames"], rx['NBatchFrame']))
            rx['Llikelihood_subframe']  = misc.my_zeros_tensor((rx["NFrames"], rx['NBatchFrame']))

            rx['net'].train()
            rx['optimiser'].zero_grad()

    rx = misc.sort_dict_by_keys(rx)

    return rx




# %%
def print_results(loss, frame, tx, fibre, rx, saving):

    
    if tx['get_time']:
        dt = time.time()-tx['get_exec_time'][1]
        tx['get_exec_time'][1] = time.time()
        tx['get_exec_time'][2][0,rx['Frame']] = dt

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
        array   = np.concatenate((Iteration, Losses, SERs),axis=1)

    if rx['Frame'] >= rx["FrameChannel"]:
        thetak = fibre['thetas'][rx['Frame']][1]+fibre['thetas'][rx['Frame']][0]
    else:
        thetak = 0


    if not tx['server']:
        if rx["mimo"].lower() == 'vae':
            if tx['flag_phase'] == 1:
                print("frame %d" % frame,
                      '--- loss     = %.1f'     % lossk,
                      '--- SNRdB    = %.2f'     % SNRdBk,
                      '--- Theta    = %.2e'     % (thetak*180/np.pi),
                      '--- std(Phi) = %.1e'     % (np.std(tx["PhaseNoise"][0, :,rx['Frame']])*180/np.pi),
                      '--- <SER>    = %.2e'     % SERmeank,
                        '--- dt       = %.2e'    % dt \
                            if tx['get_time'] \
                            else None
                      )
            else:
                print("frame %d" % frame,
                      '--- loss     = %.1f'     % lossk,
                      '--- SNRdB    = %.2f'     % SNRdBk,
                      '--- Theta    = %.2e'     % (thetak*180/np.pi),
                      '--- <SER>    = %.2e'     % SERmeank,
                        '--- dt       = %.2e'    % dt \
                            if tx['get_time'] \
                            else None
                      )
                
        else:
            if tx['flag_phase'] == 1:
                print("frame %d" % frame,
                      '--- loss     = %.3e'     % lossk,
                      '--- Theta    = %.2e'     % (thetak*180/np.pi),
                      '--- std(Phi) = %.1e'     % (np.std(tx["PhaseNoise"][0, :, rx["Frame"]])*180/np.pi),
                      '--- <SER>    = %.2e'     % SERmeank,
                        '--- dt       = %.2e'    % dt \
                            if tx['get_time'] \
                            else None
                      )
            else:
                print("frame %d" % frame,
                      '--- loss     = %.3e'     % lossk,
                      '--- Theta    = %.2e'     % (thetak*180/np.pi),
                      '--- <SER>    = %.2e'     % SERmeank,
                        '--- dt       = %.2e'    % dt \
                            if tx['get_time'] \
                            else None
                      )
    
    
    if tx['get_time']:
        dt      = np.array(tx['get_exec_time'][2][0,:rx['Frame']+1]).reshape((-1,1))
        array   = np.concatenate((array,dt),axis = 1)

    return array



#%%
def save_data(tx, fibre, rx, saving, array):

    Thetas_IN   = np.array([fibre["thetas"][k][0] for k in range(rx["NFrames"])])
    Thetas_OUT  = np.array([fibre["thetas"][k][1] for k in range(rx["NFrames"])])
    thetas      = list((Thetas_IN+Thetas_OUT)*180/np.pi)
    Thetas      = misc.list2vector(thetas)
    
    if not tx['flag_phase']:
        array2          = np.concatenate((array, Thetas), axis=1)
    elif tx['flag_phase']:
        Phis            = tx['PhaseNoise_unique'][:,rx['NBatchFrame']-1].reshape((-1,1))*180/pi
        array2          = np.concatenate((array, Thetas, Phis), axis=1)


    if rx["mimo"].lower() == "vae":
        if tx['flag_phase'] == 0:
            save_tmp = ["iteration","loss", "SNR", "SER", "Thetas"]
        else:
            save_tmp = ["iteration","loss", "SNR", "SER", "Thetas","Phis"]
            
        if 'get_exec_time' in tx:
                save_tmp.insert(4,"dt")
    else:
        if tx['flag_phase'] == 0:
            save_tmp = ["iteration","loss", "SER", "Thetas"]
        else:
            save_tmp = ["iteration","loss", "SER", "Thetas","Phis"]

        if 'get_exec_time' in tx:
                save_tmp.insert(3,"dt")
    
    misc.array2csv(array2,saving["filename"],save_tmp)

    tx      = misc.sort_dict_by_keys(tx)
    fibre   = misc.sort_dict_by_keys(fibre)
    rx      = misc.sort_dict_by_keys(rx)

    return tx, fibre, rx



