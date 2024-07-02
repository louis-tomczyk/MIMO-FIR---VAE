# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.6.1
#   Date            : 2024-07-02
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2024-03-04) - creation
#   1.0.3 (2024-04-03) - compute_loss -> compute_vae_loss
#   1.1.0 (2024-04-08) - [NEW] decision, SER_estimation,
#                            compensate_and_truncate
#   1.2.0 (2024-04-19) - decision, find_shift - storing SER values for ALL the
#                           frames
#   1.2.1 (2024-05-21) - find_shift & compensate_and_truncate - use of
#                           gen.plot_xcorr_2x2,
#                           gen.plot_decisions
#                      - decoder -> decision
#   1.3.0 (2024-05-23) - find_shift & compensate_and_truncate: handling
#                           different number of filter taps
#                      - [DELETED] phase_estimation, it will be done with
#                           matlab
#   1.4.0 (2024-05-27) - decision - including PCS: needed to scale
#                           constellations, [C2] inspired
#   1.4.1 (2024-06-06) - find_shift: NSymb_to_remove changed (no use of any
#                           weird "reference" number of taps).
#                           Ntaps -> NsampTaps
#   1.5.0 (2024-06-20) - [DRAFT] CPR_pilots: for MQAM modulations, Carrier
#                           Phase Recovery is QPSK pilot aided. see [A1]
#   1.5.1 (2024-06-21) - train_self -> train_vae, along with kit (1.1.3)
#                        decision: adapting for pilots_aided [TO BE FINISHED]
#                        [REMOVED] find_pilots
#   1.5.2 (2024-06-27) - CPR_pilots: removed decision because of synchro issues
#                           synchro ok + filtering, varargin to check
#                        compensate_and_truncate: varargin to check
#                        decision: varargin to check
#                        cleaning + "".format(...) -> f"{...}"
#   1.6.0 (2024-07-01) - mimo, SER_estimation, CPR_pilots, decision: same
#                        training schemes for cma/vae. Along with processing
#                        (1.3.0)
#   1.6.1 (2024-07-02) - CPR_pilots: adaptative filtering + Frame-FrameChannel
#                           for phase noise => Frame, along with processing
#                        (1.3.1)
#                        receiver: cleaning
#
# ----- MAIN IDEA -----
#   Library for decision functions in (optical) telecommunications
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   [A1 ]Authors        : Fabio A. Barbosa, Sandro M. Rossin Darli A.A. Mello 
#   Title               : Phase and Frequency Recovery Algorithms for
#                         Probabilistically shaped tranmission
#   Journal/Editor      : Journal of Lightwave Technology (IEEE)
#   Volume - N°         : 38 - 7
#   Date                : 2020-04-07
#   DOI/ISBN            : 10.1109/JLT.2019.2959395
#   Pages               : 1827-1835
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
#
#   [C2] Authors        : Jingtian Liu, Élie Awwad, louis Tomczyk
#       Contact         : elie.awwad@telecom-paris.fr
#       Affiliation     : Télécom Paris, COMELEC, GTO
#       Date            : 2024-04-27
#       Program Title   : 
#       Code Version    : 3.0
#       Type            : Source code
#       Web Address     : 
# ---------------------------------------------



#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt

import lib_general as gen
import lib_kit as kit
import lib_matlab as mb
import lib_maths as maths
# import tracemalloc


from lib_matlab import clc
from lib_misc import KEYS as keys
from lib_maths import get_power as power


pi = np.pi
#%% ===========================================================================
# --- CONTENTS ---
# =============================================================================
# compensate_and_truncate           --- called in : rxdsp.SER_estimation
# decision                          --- called in : rxdsp.SER_estimation
# find_shift                        --- called in : rxdsp.SER_estimation
# mimo                              --- called in : processing.processing
# SNR_estimation                    --- called in :
# =============================================================================


#%%
def receiver(tx,rx,saving):

        rx, loss        = mimo(tx, rx, saving)
        rx              = remove_symbols(rx,'data')
        rx              = CPR_pilots(tx, rx,'time trace pn')                 # {align,demod,time trace pn, time trace pn mse, trace mse}
        rx              = save_estimations(rx)
        rx              = SNR_estimation(tx, rx)
        rx              = SER_estimation(tx, rx)

        return rx, loss


    
#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================


#%%
def compensate_and_truncate(tx,rx,*varargin):

    # T == tx['Symb_real']
    THI     = rx['tmp'][0][0].round(4)
    THQ     = rx['tmp'][0][1].round(4)
    TVI     = rx['tmp'][0][2].round(4)
    TVQ     = rx['tmp'][0][3].round(4)

    # R == rx['Symb_real_dec']
    RHI     = rx['tmp'][1][0].round(4)
    RHQ     = rx['tmp'][1][1].round(4)
    RVI     = rx['tmp'][1][2].round(4)
    RVQ     = rx['tmp'][1][3].round(4)

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

    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and varargin is not None:
        gen.plot_xcorr_2x2(xcorrHI, xcorrHQ, xcorrVI, xcorrVQ,\
                           title ='compensate_and_truncate', ref = 1, zoom = 1)
    # ---------------------------------------------------------------- to check
    
    assert shiftHI == shiftHQ == shiftVI == shiftVQ == 0,\
        "shift values should be equal to 0: {} - {} - {} - {}".format(\
          shiftHI,shiftHQ,shiftVI,shiftVQ)

    tx['Symb_SER_real'] = np.zeros((tx['Npolars']*2,
                                    rx['Nframes'],
                                    rx['NSymbSER'])).astype(np.float16)
    
    rx['Symb_SER_real'] = np.zeros((tx['Npolars']*2,
                                    rx['Nframes'],
                                    rx['NSymbSER'])).astype(np.float16)


    tx['Symb_SER_real'][0,rx['Frame']]  = THI[0]
    tx['Symb_SER_real'][1,rx['Frame']]  = THQ[0]
    tx['Symb_SER_real'][2,rx['Frame']]  = TVI[0]
    tx['Symb_SER_real'][3,rx['Frame']]  = TVQ[0]

    rx['Symb_SER_real'][0,rx['Frame']]  = RHI[0]
    rx['Symb_SER_real'][1,rx['Frame']]  = RHQ[0]
    rx['Symb_SER_real'][2,rx['Frame']]  = RVI[0]
    rx['Symb_SER_real'][3,rx['Frame']]  = RVQ[0]

    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and varargin is not None:
        if rx["Frame"] == rx["Nframes"]-1:
            t = tx['Symb_SER_real'][:,rx['Frame']]
            r = rx['Symb_SER_real'][:,rx['Frame']]
            gen.plot_decisions(t, r, 15)
            # print(sum(r[0]==t[0])/len(r[0]))
            # print(sum(r[1]==t[1])/len(r[0]))
            # print(sum(r[2]==t[2])/len(r[0]))
            # print(sum(r[3]==t[3])/len(r[0]))
    # ---------------------------------------------------------------- to check

    del rx['tmp']

    return tx,rx


#%%
def CPR_pilots(tx,rx,*varargin):

    if rx['mode'].lower() == 'blind'\
    or "sig_eq_real_cma" not in rx:
        return rx

    else:
        what_pilots             = tx['pilots_info'][0]
        pilots_function         = what_pilots[0].lower()
        pilots_changes          = what_pilots[2].lower()
        tx_pilots               = tx['Symb_{}_cplx'.format(pilots_function)]

# =============================================================================
#     maintenance
# =============================================================================

        rx['NSymb_pilots_cpr']  = tx['NSymb_pilots_cpr']-tx['NSymbTaps']

        if pilots_changes != "batchwise":
            tx_pilots_H_all     = mb.repmat(tx_pilots[0], (rx['NBatchFrame'],1))
            tx_pilots_V_all     = mb.repmat(tx_pilots[1], (rx['NBatchFrame'],1))
        else:          
            tx_pilots_H_all     = tx_pilots[0]
            tx_pilots_V_all     = tx_pilots[1]

        rx_HI       = rx['sig_eq_real'][0]
        rx_HQ       = rx['sig_eq_real'][1]
        rx_VI       = rx['sig_eq_real'][2]
        rx_VQ       = rx['sig_eq_real'][3]

        rx_H        = np.expand_dims(rx_HI+1j*rx_HQ,axis = 0)
        rx_V        = np.expand_dims(rx_VI+1j*rx_VQ,axis = 0)

        rx_H_pilots = np.zeros((rx['NBatchFrame_pilots'],\
                                rx['NSymb_pilots_cpr'])).astype(np.complex64)
        rx_V_pilots = np.zeros((rx['NBatchFrame_pilots'],\
                                rx['NSymb_pilots_cpr'])).astype(np.complex64)
        
        
# =============================================================================
#     pilots alignement: sent VS received, shift caused by the shaping
# =============================================================================
        
        tx_pilots_H         = tx_pilots_H_all[1:-1]
        tx_pilots_V         = tx_pilots_V_all[1:-1]

        Nroll               = -(tx['NSymbTaps']-1)
        tx_pilots_H_roll    = np.roll(tx_pilots_H,Nroll)
        tx_pilots_V_roll    = np.roll(tx_pilots_V,Nroll)

        rx_H_rs             = np.reshape(rx_H, (rx['NBatchFrame_pilots'],-1))
        rx_V_rs             = np.reshape(rx_V, (rx['NBatchFrame_pilots'],-1))

        for k in range(int(rx['NBatchFrame_pilots'])):

            rx_H_pilots[k] = rx_H_rs[k, :rx['NSymb_pilots_cpr']]
            rx_V_pilots[k] = rx_V_rs[k, :rx['NSymb_pilots_cpr']]

    # ---------------------------------------------------------------- to check
        if len(varargin) != 0 and 'align' in varargin:
            for k in range(int(rx['NBatchFrame_pilots']/7)):            
                plt.figure()
                # 
                plt.subplot(1,2,1)
                plt.plot(np.real(tx_pilots_H_roll[k]),linewidth=2,label='TX')
                plt.plot(np.real(rx_H_pilots[k]),linewidth=2,label = 'RX')
                plt.legend()
                plt.title("polH")
                # 
                plt.subplot(1,2,2)
                plt.plot(np.real(tx_pilots_V_roll[k]),linewidth=2,label='TX')
                plt.plot(np.real(rx_V_pilots[k]),linewidth = 2,label = 'RX')
                plt.legend()
                plt.title("polV")
                # 
                plt.suptitle(f"frame = {rx['Frame']}")
                plt.show()
    # ---------------------------------------------------------------- to check
            
        del tx_pilots_H_all, tx_pilots_V_all
        del rx_HI, rx_HQ, rx_VI, rx_VQ, rx_H, rx_V
        del rx_H_rs, rx_V_rs


# =============================================================================
#       conjugaison
# =============================================================================


        tmpH = rx_H_pilots*np.conj(tx_pilots_H_roll[:,:rx['NSymb_pilots_cpr']])
        tmpV = rx_V_pilots*np.conj(tx_pilots_V_roll[:,:rx['NSymb_pilots_cpr']])
        
        tmpHn = tmpH/power(tmpH)
        tmpVn = tmpV/power(tmpV)
        
        # ------------------------------------------------------------ to check
        if len(varargin) != 0 and 'demod' in varargin:
            gen.plot_constellations(sig1 = tmpHn, sig2 = tmpVn,\
                labels= ['H','V'], title = f"cpr_pilots, frame {rx['Frame']}")
        # ------------------------------------------------------------ to check

# =============================================================================
#     phase noise: estimation
# =============================================================================
                
        phis_H          = np.angle(tmpHn)*180/pi                          # [deg]
        phis_V          = np.angle(tmpVn)*180/pi                          # [deg]
        
        phis_H_mean     = np.mean(phis_H,axis = 1)
        phis_H_std      = np.std(phis_H,axis = 1)

        phis_V_mean     = np.mean(phis_V,axis = 1)
        phis_V_std      = np.std(phis_V,axis = 1)

        del tmpHn, tmpVn, tx_pilots, rx_H_pilots, rx_V_pilots
        del tx_pilots_H, tx_pilots_V, tx_pilots_H_roll, tx_pilots_V_roll

        
# =============================================================================
#     phase noise: 
# =============================================================================
        
        rx['PhaseNoise_pilots'][rx['Frame'],0] = phis_H_mean
        rx['PhaseNoise_pilots'][rx['Frame'],1] = phis_V_mean

                                
        rx['PhaseNoise_pilots_std'][rx['Frame'],0] =phis_H_std
        rx['PhaseNoise_pilots_std'][rx['Frame'],1] =phis_V_std
    
        phi_noise_H     = rx['PhaseNoise_pilots'][rx['Frame'],0]
        phi_noise_V     = rx['PhaseNoise_pilots'][rx['Frame'],1]
        
# =============================================================================
#     noise filtering
# =============================================================================
        
        phi_noise_H_std = rx['PhaseNoise_pilots_std'][rx['Frame'],0]
        phi_noise_V_std = rx['PhaseNoise_pilots_std'][rx['Frame'],1]
        
        tmp_pn          = tx['PhaseNoise_unique'][rx['Frame'],1:-1]
        mse             = np.inf
        rea             = 0
        diff_mse        = np.inf

        pn_H_filter = maths.my_low_pass_filter(phi_noise_H, tx['pn_filt_par'])
        pn_V_filter = maths.my_low_pass_filter(phi_noise_V, tx['pn_filt_par'])
        
        # maths.plot_PSD(phi_noise_H, tx['fs']/2)
            
        while (mse > tx['pn_filt_par']['err_tolerance'])\
            and (rea < tx['pn_filt_par']['niter_max'])\
            and (tx['pn_filt_par']['window_size'] < rx['NBatchFrame']/2)\
            and (abs(diff_mse)>1e-3):
    
            pn_H_filter = maths.my_low_pass_filter(phi_noise_H, tx['pn_filt_par'])
            pn_V_filter = maths.my_low_pass_filter(phi_noise_V, tx['pn_filt_par'])

            mse_H       = maths.rmse(pn_H_filter,tmp_pn)
            mse_V       = maths.rmse(pn_V_filter,tmp_pn)
            mse         = np.round((mse_H+mse_V)/2,3)
            
            tx['pn_fil_mses'][rx['Frame'],rea] = mse
            tx['pn_filt_par']['window_size'] += 1
            
            if rea > 0:
                diff_mse = tx['pn_fil_mses'][rx['Frame'],rea-1]-mse
            
            rea += 1
            
        # ------------------------------------------------------------ to check           
            if len(varargin) != 0 and "time trace pn mse" in varargin\
                and rx['Frame'] >= rx['FrameChannel']:
                batches     = np.linspace(1,rx['NBatchFrame_pilots'],rx['NBatchFrame_pilots'])
                
                plt.figure()
                plt.plot(batches,tmp_pn*180/pi, label = 'ground truth')
                plt.plot(batches, phi_noise_H, label = 'estimation raw')
                plt.plot(batches,pn_H_filter, linestyle = 'dashed',label = "estimation filt")
                plt.legend()
                # plt.ylim(np.array([-90,90])/2)
                plt.title(f"\n frame {rx['Frame']}, mse = {mse}, window = {tx['pn_filt_par']['window_size']}")
                plt.show()
       
        if len(varargin) != 0 and "time trace pn" in varargin\
            and rx['Frame'] >= rx['FrameChannel']:
            batches     = np.linspace(1,rx['NBatchFrame_pilots'],rx['NBatchFrame_pilots'])
            
            plt.figure()
            plt.plot(batches,tmp_pn*180/pi, label = 'ground truth')
            plt.plot(batches, phi_noise_H, label = 'estimation raw')
            plt.plot(batches,pn_H_filter, linestyle = 'dashed',label = "estimation filt")
            plt.legend()
            # plt.ylim(np.array([-90,90])/2)
            plt.title(f"\n frame {rx['Frame']}, mse = {mse}, window = {tx['pn_filt_par']['window_size']}")
            plt.show()
       
        if len(varargin) != 0 and "trace mse" in varargin\
            and rx['Frame'] >= rx['FrameChannel']:
            batches     = np.linspace(1,rx['NBatchFrame_pilots'],rx['NBatchFrame_pilots'])
            
            plt.figure()
            plt.plot(tx['pn_fil_mses'][rx['Frame']])
            plt.ylim([0,4])
            plt.xlim([0,1+rx['NBatchFrame']/2])
            plt.title(f"\n frame {rx['Frame']}, mse = {mse}, window = {tx['pn_filt_par']['window_size']}")
            plt.show()
        # ------------------------------------------------------------ to check
        
        tx['pn_filt_par']['window_size'] = 1
        
# =============================================================================
#     interpolation
# =============================================================================


            

        
        return rx


#%%
def decision(tx, rx, *varargin):           

    frame   = rx['Frame']

    if rx['mode'].lower() == 'blind':
        ZHI     = np.round(np.reshape(rx["sig_eq_real"][0, frame], (1, -1)).squeeze(), 4)
        ZHQ     = np.round(np.reshape(rx["sig_eq_real"][1, frame], (1, -1)).squeeze(), 4)
        ZVI     = np.round(np.reshape(rx["sig_eq_real"][2, frame], (1, -1)).squeeze(), 4)
        ZVQ     = np.round(np.reshape(rx["sig_eq_real"][3, frame], (1, -1)).squeeze(), 4)        
        Prx     = maths.get_power(rx["sig_eq_real"][:, frame],flag_real2cplx=1,flag_flatten=1)
    
    else:
        ZHI     = np.round(np.reshape(rx["sig_eq_real"][0], (1, -1)).squeeze(), 4)
        ZHQ     = np.round(np.reshape(rx["sig_eq_real"][1], (1, -1)).squeeze(), 4)
        ZVI     = np.round(np.reshape(rx["sig_eq_real"][2], (1, -1)).squeeze(), 4)
        ZVQ     = np.round(np.reshape(rx["sig_eq_real"][3], (1, -1)).squeeze(), 4)
        Prx     = maths.get_power(rx["sig_eq_real"],flag_real2cplx=1,flag_flatten=1)
    
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
    
    Ref     = np.tile(np.unique(np.real(Xref)), [rx['NSymbEq'],1]).transpose()

    Err_HI  = abs(ZHI_ext - Ref).astype(np.float16)
    Err_HQ  = abs(ZHQ_ext - Ref).astype(np.float16)
    Err_VI  = abs(ZVI_ext - Ref).astype(np.float16)
    Err_VQ  = abs(ZVQ_ext - Ref).astype(np.float16)

    if tx["nu"] != 0: # probabilistic shaping
    
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

    ZHI_dec = tx['amps'][tmp_HI]
    ZHQ_dec = tx['amps'][tmp_HQ]
    ZVI_dec = tx['amps'][tmp_VI]
    ZVQ_dec = tx['amps'][tmp_VQ]

    rx['Symb_real_dec'][0, rx['Frame']] = ZHI_dec
    rx['Symb_real_dec'][1, rx['Frame']] = ZHQ_dec
    rx['Symb_real_dec'][2, rx['Frame']] = ZVI_dec
    rx['Symb_real_dec'][3, rx['Frame']] = ZVQ_dec
    

    
# ---------------------------------------------------------------- to check
    if len(varargin) != 0 and varargin is not None:
        # for checking the result --- time traces
        if rx["Frame"] >= rx["FrameChannel"]:
            t = [ZHI,ZHQ,ZVI,ZVQ]
            r = [ZHI_dec,ZHQ_dec,ZVI_dec,ZVQ_dec]
            gen.plot_decisions(t, r, 5)
            
        # for checking the result --- constellations
        if rx["Frame"] >= rx["FrameChannel"]:
            TX = np.reshape(tx["Symb_real"],(4,-1))
            ZX = np.array([ZHI_dec+1j*ZHQ_dec,ZVI_dec+1j*ZVQ_dec])
            gen.plot_constellations(TX,ZX, labels =['tx',"zx"])
# ---------------------------------------------------------------- to check


    return rx




#%%
def find_shift(tx,rx):

    # maintenance
    ref     = tx['Symb_real'].numpy()
    if rx['mode'].lower() != 'blind':
        tmp = ref[:,:,rx['NSymbBatch']:-rx['NSymbBatch']]
        del ref
        ref = tmp
        
    
    sig     = rx['Symb_real_dec']
    
    if ref.shape[-1] != sig.shape[-1]:
        Nsb_to_remove   = int(rx['NSymbCut_tot']-1)
        
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
    # gen.plot_xcorr_2x2(xcorrHI, xcorrHQ, xcorrVI, xcorrVQ,\
    #                   "test - find shift", ref=1, zoom=1)

    assert shiftHI == shiftHQ == shiftVI == shiftVQ,\
        "shift values should be equal :"+\
           f"{shiftHI} - {shiftHQ} - {shiftVI} - {shiftVQ}"
           
    rx['NSymbShift']    = int(shiftHI)


    return tx,rx


#%% [C1]
def mimo(tx,rx,saving,*varargin):

    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and 'b4' in varargin:
        gen.plot_constellations(sig1 = rx['sig_real'],title =
                                          f"rx b4 mimo {rx['Frame']}")
    # ---------------------------------------------------------------- to check
    
    
    
    if rx['mimo'].lower() == "vae":
        with torch.set_grad_enabled(True):
            for BatchNo in range(rx["NBatchFrame"]):

                rx              = kit.train_vae(BatchNo,rx,tx)
                rx,loss         = kit.compute_vae_loss(tx,rx)
                maths.update_fir(loss,rx['optimiser'])
    
    
    
    # ---------------------------------------------------------------- to check
        if len(varargin) != 0 and "loss" in varargin:
                gen.plot_loss_batch(rx,saving,['kind','law',"std",'linewidth'],"Llikelihood")
                gen.plot_loss_batch(rx,saving,['kind','law',"std",'linewidth'],"DKL")
                gen.plot_loss_batch(rx,saving,['kind','law',"std",'linewidth'],"losses")
    # ---------------------------------------------------------------- to check



    elif rx['mimo'].lower() == "cma" :
        rx,loss = kit.CMA(tx,rx)



    # ---------------------------------------------------------------- to check
        if len(varargin) != 0 and "loss" in varargin:
            gen.plot_loss_cma(rx,saving,['kind','law',"std",'linewidth'],"x")
    # ---------------------------------------------------------------- to check
                

    else:
        loss = []
        
        
        
    # ---------------------------------------------------------------- to check        
    if len(varargin) != 0 and "after" in varargin:
        gen.plot_constellations(rx['sig_eq_real'][:,rx['Frame'],:,:],\
                                title =f"rx after mimo {rx['Frame']}")

    if len(varargin) != 0 and "fir" in varargin:
        gen.plot_fir(rx, title =f"fir after mimo {rx['Frame']}")
    # ---------------------------------------------------------------- to check
    
    return rx,loss
    
    

#%%
def remove_symbols(rx,what):

    if rx['mimo'].lower() != 'vae':
        if what.lower() == 'data':
            if rx['mode'].lower() == 'blind':
                rx['sig_eq_real'][0] = rx['sig_eq_real_cma'][0,rx['NSymbCut']:-rx['NSymbCut']-1]
                rx['sig_eq_real'][1] = rx['sig_eq_real_cma'][1,rx['NSymbCut']:-rx['NSymbCut']-1]
                rx['sig_eq_real'][2] = rx['sig_eq_real_cma'][2,rx['NSymbCut']:-rx['NSymbCut']-1]
                rx['sig_eq_real'][3] = rx['sig_eq_real_cma'][3,rx['NSymbCut']:-rx['NSymbCut']-1]
                rx['NSymbFrame_b4Eq']= rx['NSymbFrame']-rx['NSymbCut_tot']+1
            
            else:
        
                rx['NSymbFrame_b4Eq']= rx['NSymbFrame']-2*rx['NSymbBatch']
                tmp = rx['sig_eq_real_cma'][:,rx['NSymbBatch']:-rx['NSymbBatch']-1]
                    
                # del rx['sig_eq_real']
                rx['sig_eq_real'] = tmp
                # gen.plot_constellations(rx['sig_eq_real_cma'],polar = 'H', title = "data at remove symbols")
        else:
            tmpH    = rx['{}_cplx_stuffed'.format(what)][0][rx['NSymbBatch']:-rx['NSymbBatch']]
            tmpV    = rx['{}_cplx_stuffed'.format(what)][1][rx['NSymbBatch']:-rx['NSymbBatch']]
            
            tmpH    = np.expand_dims(tmpH, axis= 0)
            tmpV    = np.expand_dims(tmpV, axis= 0)
            del rx['{}_cplx_stuffed'.format(what)]
            rx['{}_cplx_stuffed'.format(what)] = np.concatenate((tmpH,tmpV),axis = 0)
            
            # gen.plot_constellations(rx['{}_cplx_stuffed'.format(what)],polar = 'H', title = 'pilots cpr')

    return rx


# %%
def save_estimations(rx):

    rx["H_est_l"].append(rx["h_est"].tolist())

    return rx

#%%
def SER_estimation(tx,rx):

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


