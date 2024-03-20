# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2023-03-04 (1.0.0) - creation
#   Date            : 2023-03-20 - mimo {vae,cma} - CPE
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

#%%
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'


#%%
def processing_self(tx,fibre,rx,saving,flags):

    tx,fibre,rx = init_processing(tx,fibre,rx,device)
    elapsed     = np.zeros((rx['Nframes'],1))
    
    for frame in range(rx['Nframes']):
        t = time.time()
        with torch.set_grad_enabled(True):
            
            rx              = init_train(tx,rx,frame)
            tx              = txdsp.transmitter(tx,rx)
            tx,fibre,rx     = prop.propagation(tx,fibre,rx)
            rx,loss         = rxdsp.mimo(tx,rx,saving,flags)
        
        rx["H_est_l"].append(rx["h_est"].tolist())

        rx      = rxdsp.SNR_estimation(tx,rx)
        rx      = kit.CPE(rx)
        rx      = kit.SER_estimation(tx,rx)
        
        array   = print_results(loss,frame,tx,fibre,rx,saving)
        elapsed[frame] = time.time() - t
        
    tx,fibre, rx = save_data(tx,fibre,rx,saving,array)

    print('exe time = {} +/-  {}'.format(np.mean(elapsed),np.std(elapsed)))
    return tx,fibre,rx    


####################################################################################################
#%% Nested Functions
####################################################################################################





#%%
def print_results(loss,frame,tx,fibre,rx,saving):
    
    SER_valid   = rx["SER_valid"]
    SERs        = rx["SERs"]
    Losses      = rx["Losses"]

    if rx["mimo"].lower() == "vae":
        SNRdB_est   = rx["SNRdB_est"][frame].item()  
        SNRdBs      = rx["SNRdBs"]
        lossk       = round(loss.item(),0)
        SNRdBk      = round(10*np.log10(SNRdB_est),5)
        SNRdBs.append(SNRdBk)
        SNRdBs      = misc.list2vector(SNRdBs)
        SERvalid0   = round(SER_valid[0,frame].item(),15)
        SERvalid1   = round(SER_valid[1,frame].item(),15)
        SERvalid2   = round(SER_valid[2,frame].item(),15)
        SERvalid3   = round(SER_valid[3,frame].item(),15)
        
        SERsvalid   = np.array([SERvalid0,SERvalid1,SERvalid2,SERvalid3])
        SERmeank    = round(np.mean(SERsvalid),15)
    
        Shifts      = np.array([rx['shift'][0].item(),rx['shift'][1].item()])
        Shiftsmean  = round(np.mean(Shifts),1)
        
        SERs.append(SERmeank)
        SERs        = misc.list2vector(SERs)

        
    else:
        if rx['Frame']>rx["FrameRndRot"]:
            lossk   = torch.std(loss[-1][:]).item()
        else:
            lossk   = np.NaN


    Losses.append(lossk)

    
    Iteration   = misc.linspace(1,len(Losses),len(Losses))
    Losses      = misc.list2vector(Losses)
    
    if rx['mimo'].lower() == "vae":
        array   = np.concatenate((Iteration,Losses,SNRdBs,SERs),axis=1)
    else:
        # array   = np.concatenate((Iteration,Losses,SERs),axis=1)
        array   = np.concatenate((Iteration,Losses),axis=1)
        
    thetak      = fibre['thetas'][frame][1]+fibre['thetas'][frame][0]
        
    if rx["mimo"].lower() == 'vae':
        print("frame %d"        % frame,
          '--- loss = %.1f'     % lossk,
          '--- SNRdB = %.2f'    % SNRdBk,
          '--- Theta = %.2f'    % (thetak*180/np.pi),
          '--- std(Phi) = %.1f' % (np.std(tx["PhaseNoise"][0,:,rx["Frame"]])*180/np.pi),
          '--- <SER> = %.2e'    % SERmeank,
          # '--- r = %i'          % rx['r']
          )
    else:
        print("frame %d"        % frame,
          '--- loss = %.3e'     % lossk,
          # '--- <SER> = %.2e'    % SERmeank,
          )    
    
    
    '''
        shift_x = %.1f' % shift[0].item()
        shift_y = %.1f' % shift[1].item(),
        
        (constell. with shaping)
        SER_x = %.3e '  % SERvalid0,
        SER_y = %.3e'   % SERvalid1,
        
        (soft demapper)
        SER_x = %.3e '  % SERvalid2,
        SER_y = %.3e'   % SERvalid3,
    '''
    
    return array
# ============================================================================================ #

#%%
def save_data(tx,fibre,rx,saving,array):
    
    
    Thetas_IN   = np.array([fibre["thetas"][k][0] for k in range(rx["Nframes"])])
    Thetas_OUT  = np.array([fibre["thetas"][k][1] for k in range(rx["Nframes"])])
    thetas      = list((Thetas_IN+Thetas_OUT)*180/np.pi)
    Thetas      = misc.list2vector(thetas)
    array2      = np.concatenate((array,Thetas),axis=1)
    
    if rx["mimo"].lower() == "vae":
        misc.array2csv(array2, saving["filename"],["iteration","loss","SNR","SER","Thetas"])
    else:
        misc.array2csv(array2, saving["filename"],["iteration","loss","Thetas"])

    tx          = misc.sort_dict_by_keys(tx)
    fibre       = misc.sort_dict_by_keys(fibre)            
    rx          = misc.sort_dict_by_keys(rx)

    return tx,fibre,rx
# ============================================================================================ #

#%%
def init_processing(tx,fibre,rx,device):
    tx['Npolars']           = 2
    tx["Ntaps"]             = tx["Nsps"]*tx["NSymbTaps"]-1
    tx["RollOff"]           = 0.1
    tx['NSymbFrame']        = rx["NSymbFrame"]
    # Nconv = NSymbFrame + some extra symbols necessary for edges management
    tx["Nconv"]             = rx["NSymbFrame"]+tx["Ntaps"]+1
    tx["Nsamp_up"]          = tx["Nsps"]*(tx["Nconv"]-1)+1
    tx["Nsamp_gross"]       = tx["Nsamp_up"]-(tx["Ntaps"]-1)
    
    # Channel matrix initiatlisation: Dirac
    h_est                           = np.zeros([tx['Npolars'],tx['Npolars'],2,tx["Ntaps"]])
    h_est[0,0,0,tx["NSymbTaps"]-1]  = 1
    h_est[1,1,0,tx["NSymbTaps"]-1]  = 1
    h_est                           = misc.my_tensor(h_est,requires_grad=True)
 
    tx, rx                  = txdsp.get_constellation(tx,rx)
    SNR                     = 10**(rx["SNRdB"]/10)
    rx["noise_var"]         = torch.full((2,),tx['pow_mean']/SNR/2)
    
    tx["fs"]                = tx["Rs"]*tx["Nsps"]           # [GHz]
    
    rx["h_est"]             = h_est
    rx["NBatchFrame"]       = np.floor(rx['NSymbFrame']/rx['BatchLen']).astype("int")
    rx["NsampBatch"]        = rx['BatchLen']* tx['Nsps']
    
    rx["N_cut"]             = 3# number of symbols cut off to prevent edge effects of convolution
    rx["Nsamp_rx"]          = tx["Nsps"]*rx["NSymbFrame"]
    
    
    fibre                   = prop.set_thetas(tx,fibre,rx)
    tx                      = txhw.gen_phase_noise(tx,rx)
    # fibre['phiIQ']          = np.array([0, 0], dtype=np.complex64)
    
    rx['net']               = kit.twoXtwoFIR(tx).to(device)
    rx['optimiser']         = optim.Adam(rx['net'].parameters(), lr = rx['lr'])
    rx['optimiser'].add_param_group({"params": rx["h_est"]}) 
    
    # initialisation of the outputs    
    if rx["mimo"].lower() == "cma":
        rx["out_cpe"] = dict()
        for k in range(rx["Nframes"]-rx['FrameRndRot']):
            rx['out_cpe'][str(k)] = dict()
            
    rx["H_est_l"]           = []
    rx["H_lins"]            = []
    
    rx["Losses"]            = []
    rx["SNRdBs"]            = []
    rx["SNRdB_est"]         = np.zeros(rx['Nframes'])
    rx["SER_valid"]         = []
    rx["SERs"]              = []
    
    rx["SER_valid"]         = misc.my_zeros_tensor((4,rx['Nframes']))
    rx['Pnoise_est']        = misc.my_zeros_tensor((tx["Npolars"],rx['Nframes']))
    rx["minibatch_real"]    = misc.my_zeros_tensor((tx["Npolars"],2,rx["NsampBatch"]))
    
    tx      = misc.sort_dict_by_keys(tx)
    fibre   = misc.sort_dict_by_keys(fibre)
    rx      = misc.sort_dict_by_keys(rx)
    
    return tx,fibre,rx
# ============================================================================================ #


#%%
def init_train(tx,rx,frame):
    
    if rx["N_lrhalf"] > rx["Nframes"]:
        
        rx["lr_scheduled"]              = rx["lr"] * 0.5
        rx['optimiser'].param_groups[0]['lr'] = rx["lr_scheduled"]
        
    rx['out_train']         = misc.my_zeros_tensor((tx["Npolars"],2*tx["N_amps"],rx["NSymbFrame"]))
    
        
    rx['Pnoise_batches']    = misc.my_zeros_tensor((tx["Npolars"],rx["NBatchFrame"]))
    rx["Frame"]             = frame
    
    if rx["mimo"].lower() == 'vae':
        rx['out_const']             = misc.my_zeros_tensor((tx["Npolars"],2,rx["NSymbFrame"]))
        rx['losses_subframe']       = misc.my_zeros_tensor((rx["Nframes"],rx['NBatchFrame']))
        rx['DKL_subframe']          = misc.my_zeros_tensor((rx["Nframes"],rx['NBatchFrame']))
        rx['Llikelihood_subframe']  = misc.my_zeros_tensor((rx["Nframes"],rx['NBatchFrame']))
    
    
    rx['net'].train()
    rx['optimiser'].zero_grad()
    
    return rx
# ============================================================================================ #
