# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2024-03-04 (1.0.0)    creation
#                   : 2024-04-03 (1.0.3)    compute_loss -> compute_vae_loss
#                   : 2024-04-08 (1.1.0)    [NEW] decision, SER_estimation, compensate_and_truncate
#                   : 2024-04-19 (1.2.0)    decision, find_shift --- storing SER values for ALL the frames
#                   : 2024-05-21 (1.2.1)    find_shift & compensate_and_truncate --- use of 
#                                               gen.plot_xcorr_2x2, gen.plot_decisions
#                   :                       decoder -> decision
#                   : 2024-05-23 (1.3.0)    find_shift & compensate_and_truncate: handling different number of filter taps
#                                           [DELETED] phase_estimation, it will be done with matlab
#   Date            : 2024-05-27 (1.4.0)    decision --- including PCS: needed to scale constellations, inspired from [C2]
#   Version         : 1.3.0
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute - place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source - include copyright - state changes - include license
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books
#   Authors             :
#   Title               :
#   Jounal/Editor       :
#   Volume - N°         :
#   Date                :
#   DOI/ISBN            :
#   Pages               :
#  ----------------------
#   Functions           :
#   Author              : [C1] Vincent LAUINGER
#   Author contact      : vincent.lauinger@kit.edu
#   Affiliation         : Communications Engineering Lab (CEL)
#                           Karlsruhe Institute of Technology (KIT)
#   Date                : 2022-06-15
#   Title of program    :
#   Code version        :
#   Web Address         : https://github.com/kit-cel/vae-equalizer
#  ----------------------
#   Functions           :
#   Author              : [C2] Jingtian LIU, Élie AWWAD, louis TOMCZYK
#   Author contact      : elie.awwad@telecom-paris.fr
#   Affiliation         : Télécom Paris, COMELEC, GTO
#   Date                : 2024-04-27
#   Title of program    : 
#   Code version        : 3.0
#   Type                : source code
#   Web Address         :
# ---------------------------------------------


#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================

import numpy as np
import torch

import matplotlib.pyplot as plt
import lib_kit as kit
import lib_maths as maths
import lib_general as gen
import lib_rxdsp as rxdsp


#%% ===========================================================================
# --- CONTENTS ---
# =============================================================================
# compensate_and_truncate           --- called in : rxdsp.SER_estimation
# decision                           --- called in : rxdsp.SER_estimation
# find_shift                        --- called in : rxdsp.SER_estimation
# mimo                              --- called in : processing.processing
# SNR_estimation                    --- called in :
# =============================================================================

    
#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================


#%%
def compensate_and_truncate(tx,rx):

    # T == tx['Symb_real']
    THI     = rx['tmp'][0][0]
    THQ     = rx['tmp'][0][1]
    TVI     = rx['tmp'][0][2]
    TVQ     = rx['tmp'][0][3]

    # R == rx['Symb_real_dec']
    RHI     = rx['tmp'][1][0]
    RHQ     = rx['tmp'][1][1]
    RVI     = rx['tmp'][1][2]
    RVQ     = rx['tmp'][1][3]

    if rx["mimo"].lower() == "cma":
        rx['NSymbSER']  = len(THI[0])
    else:
        rx['NSymbSER']  = rx["NSymbFrame"]

    # correcting the shift
    RHI     = np.roll(RHI,rx['NSymbShift'])
    RHQ     = np.roll(RHQ,rx['NSymbShift'])
    RVI     = np.roll(RVI,rx['NSymbShift'])
    RVQ     = np.roll(RVQ,rx['NSymbShift'])

    # correlation between each channel
    xcorrHI = np.correlate(THI[0],RHI[0], 'same')
    xcorrHQ = np.correlate(THQ[0],RHQ[0], 'same')
    xcorrVI = np.correlate(TVI[0],RVI[0], 'same')
    xcorrVQ = np.correlate(TVQ[0],RVQ[0], 'same')

    # getting the shift
    shiftHI = np.argmax(xcorrHI)-rx["NSymbEq"]/2
    shiftHQ = np.argmax(xcorrHQ)-rx["NSymbEq"]/2
    shiftVI = np.argmax(xcorrVI)-rx["NSymbEq"]/2
    shiftVQ = np.argmax(xcorrVQ)-rx["NSymbEq"]/2

    # displaying correlations if needed to check
    # gen.plot_xcorr_2x2(xcorrHI, xcorrHQ, xcorrVI, xcorrVQ,title ='test - compensated', ref = 1, zoom = 1)

    assert shiftHI == shiftHQ == shiftVI == shiftVQ == 0,\
        "shift values should be equal to 0: {} - {} - {} - {}".format(\
          shiftHI,shiftHQ,shiftVI,shiftVQ)

    tx['Symb_SER_real'] = np.zeros((tx['Npolars']*2,rx['Nframes'],rx['NSymbSER'])).astype(np.float16)
    rx['Symb_SER_real'] = np.zeros((tx['Npolars']*2,rx['Nframes'],rx['NSymbSER'])).astype(np.float16)


    tx['Symb_SER_real'][0,rx['Frame']]  = THI[0]
    tx['Symb_SER_real'][1,rx['Frame']]  = THQ[0]
    tx['Symb_SER_real'][2,rx['Frame']]  = TVI[0]
    tx['Symb_SER_real'][3,rx['Frame']]  = TVQ[0]

    rx['Symb_SER_real'][0,rx['Frame']]  = RHI[0]
    rx['Symb_SER_real'][1,rx['Frame']]  = RHQ[0]
    rx['Symb_SER_real'][2,rx['Frame']]  = RVI[0]
    rx['Symb_SER_real'][3,rx['Frame']]  = RVQ[0]

    '''
    # displaying signals if needed to check
    if rx["Frame"] == rx["Nframes"]-1:
        t = tx['Symb_SER_real'][:,rx['Frame']]
        r = rx['Symb_SER_real'][:,rx['Frame']]
        gen.plot_decisions(t, r, 15)
    '''

    del rx['tmp']

    return tx,rx


#%%
def decision(tx, rx):           
            
    if rx['mimo'].lower() == "cma" and rx['Frame'] < rx['FrameChannel']:
        return rx
    
    else:

        if rx['mimo'].lower() == "cma":
            frame   = rx['Frame']-rx['FrameChannel']
        else:
            frame   = rx['Frame']


        ZHI     = np.round(np.reshape(rx["sig_eq_real"][0, frame], (1, -1)).squeeze(), 4)
        ZHQ     = np.round(np.reshape(rx["sig_eq_real"][1, frame], (1, -1)).squeeze(), 4)
        ZVI     = np.round(np.reshape(rx["sig_eq_real"][2, frame], (1, -1)).squeeze(), 4)
        ZVQ     = np.round(np.reshape(rx["sig_eq_real"][3, frame], (1, -1)).squeeze(), 4)
        
        Prx     = maths.get_power(rx["sig_eq_real"][:, frame],flag_real2cplx=1,flag_flatten=1)
        
        # (1.3.0): no division by the power
        ZHI     = ZHI/np.sqrt(Prx[0])
        ZHQ     = ZHQ/np.sqrt(Prx[0])
        ZVI     = ZVI/np.sqrt(Prx[1])
        ZVQ     = ZVQ/np.sqrt(Prx[1])
        
        # checking normalisation
        # print(maths.get_power(np.array([ZHI+1j*ZHQ,ZVI+1j*ZVQ]))) # should equal 1
        
        M       = int(tx['mod'][0:-3])                  # 'M' for M-QAM
        ZHI_ext = np.tile(ZHI, [int(np.sqrt(M)), 1])
        ZHQ_ext = np.tile(ZHQ, [int(np.sqrt(M)), 1])
        ZVI_ext = np.tile(ZVI, [int(np.sqrt(M)), 1])
        ZVQ_ext = np.tile(ZVQ, [int(np.sqrt(M)), 1])

        # # (1.4.0)
        X_alphabet  = tx['const_affixes']
        Px_alphabet = maths.get_power(X_alphabet)
        Xref        = X_alphabet/np.sqrt(Px_alphabet)
        Ptx         = maths.get_power(np.reshape(tx["Symb_real"],(4,-1)),flag_real2cplx=1)
        
        if tx['nu'] != 0:
            Xref = Xref/np.sqrt(np.mean(Ptx))
        
        # (1.3.0)         
        # Ref     = np.tile(tx['amps'].unsqueeze(1), [1, rx['NSymbEq']])
        # (1.4.0)
        Ref     = np.tile(np.unique(np.real(Xref)), [rx['NSymbEq'],1]).transpose()
        
        
        Err_HI = abs(ZHI_ext - Ref).astype(np.float16)
        Err_HQ = abs(ZHQ_ext - Ref).astype(np.float16)
        Err_VI = abs(ZVI_ext - Ref).astype(np.float16)
        Err_VQ = abs(ZVQ_ext - Ref).astype(np.float16)

        if tx["nu"] != 0: # uniform constellation
        
            if rx['mimo'].lower() != "vae" and type(tx["prob_amps"]) != torch.Tensor:
                tx["prob_amps"]         = torch.tensor(tx["prob_amps"])

            prob_amps   = np.tile(tx['prob_amps'].unsqueeze(1), [1, rx['NSymbEq']])
            SNR         = 10**(rx['SNRdB']/10)

            Err_HI      = Err_HI ** 2 - 1/SNR*np.log(prob_amps)
            Err_HQ      = Err_HQ ** 2 - 1/SNR*np.log(prob_amps)
            Err_VI      = Err_VI ** 2 - 1/SNR*np.log(prob_amps)
            Err_VQ      = Err_VQ ** 2 - 1/SNR*np.log(prob_amps)

        tmp_HI  = np.argmin(Err_HI, axis=0)
        tmp_HQ  = np.argmin(Err_HQ, axis=0)
        tmp_VI  = np.argmin(Err_VI, axis=0)
        tmp_VQ  = np.argmin(Err_VQ, axis=0)
    
        ZHI_dec = tx['amps'][tmp_HI]#.numpy()
        ZHQ_dec = tx['amps'][tmp_HQ]#.numpy()
        ZVI_dec = tx['amps'][tmp_VI]#.numpy()
        ZVQ_dec = tx['amps'][tmp_VQ]#.numpy()

        rx['Symb_real_dec'][0, rx['Frame']] = ZHI_dec
        rx['Symb_real_dec'][1, rx['Frame']] = ZHQ_dec
        rx['Symb_real_dec'][2, rx['Frame']] = ZVI_dec
        rx['Symb_real_dec'][3, rx['Frame']] = ZVQ_dec
        

        
        '''
        # for checking the result --- time traces
        if rx["Frame"] => rx["FrameChannel"]:
            t = [ZHI,ZHQ,ZVI,ZVQ]
            r = [ZHI_dec,ZHQ_dec,ZVI_dec,ZVQ_dec]
            gen.plot_decisions(t, r, 5)
            
        # for checking the result --- constellations
        if rx["Frame"] => rx["FrameChannel"]:
            TX = np.reshape(tx["Symb_real"],(4,-1))
            ZX = np.array([ZHI_dec+1j*ZHQ_dec,ZVI_dec+1j*ZVQ_dec])
            gen.plot_const_2pol_2sig(TX, ZX, ['tx',"zx"])
        '''

    return rx



#%%
def find_shift(tx,rx):

    # maintenance
    ref     = tx['Symb_real'].numpy()
    sig     = rx['Symb_real_dec']
    
    if ref.shape[-1] != sig.shape[-1]:
        Nsb_taps_ref    = 7
        Nsb_to_remove   = int((Nsb_taps_ref - (tx["NSymbTaps"]-Nsb_taps_ref)/2)*tx["Nsps"])
        
        ref = ref[:,:,Nsb_to_remove:]
        # print('|ref.shape - sig.shape| = ',ref.shape[-1]-sig.shape[-1])

    THI     = np.expand_dims(ref[0,0,:],0)
    THQ     = np.expand_dims(ref[0,1,:],0)
    TVI     = np.expand_dims(ref[1,0,:],0)
    TVQ     = np.expand_dims(ref[1,1,:],0)

    RHI     = np.reshape(sig[0,rx['Frame']],(1,-1))
    RHQ     = np.reshape(sig[1,rx['Frame']],(1,-1))
    RVI     = np.reshape(sig[2,rx['Frame']],(1,-1))
    RVQ     = np.reshape(sig[3,rx['Frame']],(1,-1))

    rx['tmp'] = [[THI,THQ,TVI,TVQ],[RHI,RHQ,RVI,RVQ]]

    # correlation between each channel
    xcorrHI = np.correlate(THI[0],RHI[0], 'same')
    xcorrHQ = np.correlate(THQ[0],RHQ[0], 'same')
    xcorrVI = np.correlate(TVI[0],RVI[0], 'same')
    xcorrVQ = np.correlate(TVQ[0],RVQ[0], 'same')

    # getting the shift
    shiftHI = np.argmax(xcorrHI)-rx["NSymbEq"]/2
    shiftHQ = np.argmax(xcorrHQ)-rx["NSymbEq"]/2
    shiftVI = np.argmax(xcorrVI)-rx["NSymbEq"]/2
    shiftVQ = np.argmax(xcorrVQ)-rx["NSymbEq"]/2

    # displaying the correaltions to check
    # gen.plot_xcorr_2x2(xcorrHI, xcorrHQ, xcorrVI, xcorrVQ, "test - find shift", ref=1, zoom=1)

    assert shiftHI == shiftHQ == shiftVI == shiftVQ, "shift values should be equal : {} - {} - {} - {}".format(shiftHI,shiftHQ,shiftVI,shiftVQ)
    rx['NSymbShift']    = int(shiftHI)


    return tx,rx


#%%
def mimo(tx,rx,saving,flags):

    if rx['mimo'].lower() == "vae":
        with torch.set_grad_enabled(True):
            for BatchNo in range(rx["NBatchFrame"]):

                rx              = kit.train_self(BatchNo,rx,tx)
                rx,loss         = kit.compute_vae_loss(tx,rx)
                maths.update_fir(loss,rx['optimiser'])
                        
                # if rx['Frame'] >= rx['FrameChannel']:
                #     if BatchNo%40 == 0:
        gen.plot_const_2pol(rx['sig_real'], "RX f-{} B-{}".format(rx['Frame'],BatchNo))
            
            # plot_loss_batch(rx,flags,saving,['kind','law',"std",'linewidth'],"Llikelihood")
            # plot_loss_batch(rx,flags,saving,['kind','law',"std",'linewidth'],"DKL")
            # plot_loss_batch(rx,flags,saving,['kind','law',"std",'linewidth'],"losses")

        return rx,loss

    elif rx['mimo'].lower() == "cma" :

        if rx["Frame"]>rx['FrameChannel']-1:
            rx,loss = kit.CMA(tx,rx)
            # gen.plot_loss_cma(rx,flags,saving,['kind','law',"std",'linewidth'],"x")
            gen.plot_const_2pol(rx['sig_eq_real'], "RX f-{}".format(rx['Frame']))
        else:
            loss = []

        return rx,loss

    else:
        loss = []
        return rx,loss

    # if rx['Frame'] >= rx['FrameChannel']:
    #     title = "eq f-{}".format(rx['Frame']-rx['FrameChannel'])
    #     gen.plot_const_2pol(rx['sig_eq_real'], title, tx)



#%%
def SER_estimation(tx,rx):

    if rx['mimo'].lower() == "cma" and rx['Frame'] < rx['FrameChannel']:

        return rx

    else:
        rx      = decision(tx,rx)
        tx,rx   = find_shift(tx, rx)
        tx,rx   = compensate_and_truncate(tx,rx)

        RH      = rx['Symb_SER_real'][0,rx['Frame']]+1j*rx['Symb_SER_real'][1,rx['Frame']]
        RV      = rx['Symb_SER_real'][2,rx['Frame']]+1j*rx['Symb_SER_real'][3,rx['Frame']]

        TH      = tx['Symb_SER_real'][0,rx['Frame']]+1j*tx['Symb_SER_real'][1,rx['Frame']]
        TV      = tx['Symb_SER_real'][2,rx['Frame']]+1j*tx['Symb_SER_real'][3,rx['Frame']]


        rx["SER_valid"][0,rx['Frame']]  = sum(RH!=TH)/rx['NSymbSER']
        rx["SER_valid"][1,rx['Frame']]  = sum(RV!=TV)/rx['NSymbSER']

        return rx


#%% [C1]
def SNR_estimation(tx,rx):

    if rx["mimo"].lower() == "vae":
        rx["SNRdB_est"][rx['Frame']]      = tx["pow_mean"]/torch.mean(rx['Pnoise_batches'])
        rx['Pnoise_est'][:,rx['Frame']]   = torch.mean(rx['Pnoise_batches'],dim=1)  

    return rx



