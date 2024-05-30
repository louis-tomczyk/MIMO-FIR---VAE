# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2023-03-04 (1.0.0)    creation
#   Date            : 2024-03-27 (1.1.0)    print_result --- PhaseNoise management
#                                           [NEW] save estimations
#                   : 2024-04-03 (1.1.1)    print_result
#                   : 2024-04-19 (1.2.0)    init_processing, along with rxdsp (1.1.0)
#                   : 2024-05-22 (1.2.1)    init_processing, along with rxdsp (1.3.0)
#                                           init_processing, print_result --- diffentiating VAE-CMA for initialisation
#   Date            : 2024-05-28 (1.2.2)    print_result
#   Version         : 1.2.2
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute - place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source - include copyright - state changes - include license
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books
#   Authors             : [A1] Vincent LAUINGER
#   Title               : Blind Equalization and Channel Estimation in Coherent Optical
#                         Communications Using Variational Autoencoders
#   Jounal/Editor       : JOURNAL ON SELECTED AREAS IN COMMUNICATIONS
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
import timeit

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================

#%%
def processing(tx, fibre, rx, saving, flags):

    tx, fibre, rx       = init_processing(tx, fibre, rx, saving, device)

    for frame in range(rx['Nframes']):

        # print(frame)

        rx              = init_train(tx, rx, frame)
        tx              = txdsp.transmitter(tx, rx)
        tx, fibre, rx   = prop.propagation(tx, fibre, rx)


        rx, loss        = rxdsp.mimo(tx, rx, saving, flags)

        rx              = save_estimations(rx)
        rx              = rxdsp.SNR_estimation(tx, rx)
        rx              = rxdsp.SER_estimation(tx, rx)
        array           = print_results(loss, frame, tx, fibre, rx, saving)

        # rx = gen.real2complex_fir(rx)
        # gen.show_fir_central_tap(rx)
        # gen.show_pol_matrix_central_tap(fibre, rx)



    tx, fibre, rx = save_data(tx, fibre, rx, saving, array)

    return tx, fibre, rx



# %%

def init_processing(tx, fibre, rx, saving, device):

# -----------------------------------------------------------------------------
# MISSING FIELDS
# -----------------------------------------------------------------------------


    if "channel" not in fibre:
        fibre['channel']    = "linear"
    

    if 'shaping_filter' not in tx and fibre["channel"].lower() != "awgn":
        tx['shaping_filter']= "rrc"

    if 'skip' not in saving:
        saving['skip']      = 10


# -----------------------------------------------------------------------------
# SYMBOLS AND SAMPLES MANAGEMENT
# -----------------------------------------------------------------------------

    # ----------------------------------
    
    tx['NSymbTot']          = tx['NSymbFrame']*rx['Nframes']
    tx['NSymbFrame']        = rx["NSymbFrame"]
    
    rx["NBatchFrame"]       = int(rx['NSymbFrame']/rx['NSymbBatch'])
    rx["NSymbCut"]          = 10  # number of symbols cut off to prevent edge effects of convolution
    rx["NSymbCut_tot"]      = 2*rx['NSymbCut']+1
    
    rx['NSymbEq']       = rx["NSymbFrame"]    
    if rx['mimo'].lower() != "vae":        
        rx['NSymbEq']       += tx['NSymbTaps']-rx['NSymbCut_tot']
        
    # ----------------------------------

    tx["Ntaps"]             = tx["Nsps"]*tx["NSymbTaps"]-1      # length of FIR filter
    tx["Nconv"]             = rx["NSymbFrame"]+tx["Ntaps"]+1    # some extra symbols necessary for edges management
    
    tx["Nsamp_up"]          = tx["Nsps"]*(tx["Nconv"]-1)+1
    tx["NsampFrame"]        = tx["Nsamp_up"]-(tx["Ntaps"]-1)
    tx['NsampTot']          = tx['NsampFrame']*rx['Nframes']
    rx["NsampBatch"]        = rx['NSymbBatch']*tx['Nsps']    
    rx['NsampFrame']        = rx["NsampBatch"]*rx['NBatchFrame']

    rx["NframesChannel"]    = rx["Nframes"]-rx["FrameChannel"]   
    rx["NframesTraining"]   = rx["Nframes"]-rx["NframesChannel"]
     
    rx["NBatchesChannel"]   = rx["NframesChannel"]*rx['NBatchFrame'] 
    rx["NBatchesTot"]       = rx["Nframes"]*rx['NBatchFrame']
    rx["NBatchesTraining"]  = rx["NBatchesTot"]-rx['NBatchesChannel']

    
    tx['NsampChannel']      = tx['NsampFrame']*rx['NframesChannel']
    tx['Nsamptraining']     = tx['NsampTot']-tx['NsampChannel'] 



# -----------------------------------------------------------------------------
# TRANSMITTER
# -----------------------------------------------------------------------------

    tx['Npolars']           = 2
    tx["RollOff"]           = 0.1
    tx["fs"]                = tx["Rs"]*tx["Nsps"]           # [GHz]
    tx, rx                  = txdsp.get_constellation(tx, rx)
    # tx["PhaseNoise"]        = np.zeros((tx['Npolars'],tx['NsampTot']))
    # tx["PhaseNoise"]        = np.zeros((tx['Npolars'],tx['NsampChannel'], rx['NframesChannel']))
    tx["PhaseNoise"]        = np.zeros((tx['Npolars'],tx['NsampFrame'], rx['Nframes']))
# -----------------------------------------------------------------------------
# INITIALISATION OF CHANNEL MATRIX
# -----------------------------------------------------------------------------

    h_est                   = np.zeros([tx['Npolars'], tx['Npolars'], 2, tx["Ntaps"]])
    h_est[0,0,0,tx["NSymbTaps"]-1] = 1
    h_est[1,1,0,tx["NSymbTaps"]-1] = 1
    
    if rx['mimo'].lower() == "vae":
        h_est               = misc.my_tensor(h_est, requires_grad=True)
        
    rx["h_est"]             = h_est


    fibre['phiIQ']          = np.zeros(tx['Npolars'],dtype = complex)
    
    rx['SNR']               = 10**(rx['SNRdB']/10)
    rx["noise_var"]         = torch.full((2,), tx['pow_mean']/rx['SNR']/2)



# -----------------------------------------------------------------------------
# RANDOM EFFECTS
# -----------------------------------------------------------------------------
    fibre                   = prop.set_thetas(tx, fibre, rx)
    tx                      = txhw.gen_phase_noise(tx, rx)
    # tx                      = txhw.set_phis(tx, rx)

# -----------------------------------------------------------------------------
# RANDOM EFFECTS
# -----------------------------------------------------------------------------

    if rx['mimo'].lower() == "vae":
        rx['net']               = kit.twoXtwoFIR(tx).to(device)
        rx['optimiser']         = optim.Adam(rx['net'].parameters(), lr=rx['lr'])
        rx['optimiser'].add_param_group({"params": rx["h_est"]})

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------

    # OUTPUTS
    # if rx["mimo"].lower() == "cma":
    #     rx["out_cpe"] = dict()
    #     for k in range(rx["Nframes"]-rx['FrameChannel']):
    #         rx['out_cpe'][str(k)] = dict()


    rx["H_est_l"]           = []
    rx["H_lins"]            = []
    rx["Losses"]            = []
    rx["SNRdBs"]            = []
    rx["SERs"]              = []


    rx["SNRdB_est"]         = np.zeros(rx['Nframes'])
    rx['sig_real']          = torch.zeros((tx['Npolars']*2,tx['NsampFrame']))


    if rx['mimo'].lower() == "vae":
        rx['sig_eq_real']   = np.zeros((tx['Npolars']*2, rx['Nframes'], rx['NBatchFrame'],rx['NSymbBatch'])).astype(np.float32)

    else:
        rx['sig_eq_real']   = np.zeros((tx['Npolars']*2, rx['NframesChannel'], rx['NSymbEq'])).astype(np.float32)


    rx['Symb_real_dec'] = np.zeros((tx['Npolars']*2,rx['Nframes'],rx['NSymbEq']),dtype = np.float16)
    rx["SER_valid"]     = np.zeros((2, rx['Nframes']),dtype = np.float128)
    rx['Pnoise_est']    = np.zeros((tx["Npolars"], rx['Nframes']),dtype = np.float32)


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
            rx['Pnoise_batches']        = misc.my_zeros_tensor((tx["Npolars"], rx["NBatchFrame"]))
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
        # SERvalid2       = round(SER_valid[2, frame].item(), 15)
        # SERvalid3       = round(SER_valid[3, frame].item(), 15)

        # SERsvalid       = np.array([SERvalid0, SERvalid1, SERvalid2, SERvalid3])
        SERsvalid       = np.array([SERvalid0, SERvalid1])
        SERmeank        = round(np.mean(SERsvalid), 15)


    else:
        if rx['Frame'] >= rx["FrameChannel"]:
            # frame       = frame - rx['FrameChannel']
            
            if rx['mimo'].lower() == "vae":
                lossk   = torch.std(loss[-1][:]).item()
            else:
                lossk   = np.std(loss[-1][:]).item()
                
            # SERvalid0   = round(SER_valid[0, frame].item(), 15)
            # SERvalid1   = round(SER_valid[1, frame].item(), 15)
            # SERvalid2   = round(SER_valid[2, frame].item(), 15)
            # SERvalid3   = round(SER_valid[3, frame].item(), 15)
            
            SERvalidH   = round(SER_valid[0, rx['Frame']].item(), 15)
            SERvalidV   = round(SER_valid[1, rx['Frame']].item(), 15)

            SERsvalid   = np.array([SERvalidH, SERvalidV])
            SERmeank    = round(np.mean(SERsvalid), 15)

        else:
            lossk       = np.NaN
            SERmeank    = np.NaN



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
                  # '--- std(Phi) = %.1f'     % (np.std(tx["PhaseNoise"][0, :, rx["Frame"]])*180/np.pi),
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
        if rx['Frame'] >= rx["FrameChannel"]:
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


# %%
def save_estimations(rx):

    rx["H_est_l"].append(rx["h_est"].tolist())

    # if rx['Frame'] >= rx['FrameChannel']:
    #     rx["phases_est_mat"][rx["Frame"],:] = rx["phases_est"].squeeze()

    return rx
