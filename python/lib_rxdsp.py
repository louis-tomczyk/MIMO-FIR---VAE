# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.7.3
#   Date            : 2024-07-10
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
#                           plot.xcorr_2x2,
#                           plot.decisions
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
#   1.5.0 (2024-06-20) - [NEW] remove_symbols
#                        [DRAFT] CPR_pilots: for MQAM modulations, Carrier
#                           Phase Recovery is QPSK pilot aided. see [A1]
#   1.5.1 (2024-06-21) - train_self -> train_vae, along with kit (1.1.3)
#                        decision: adapting for pilots_aided [TO BE FINISHED]
#                        [REMOVED] find_pilots
#   1.5.2 (2024-06-27) - CPR_pilots: removed decision because of synchro issues
#                           synchro ok + filtering, varargin to check
#                        compensate_and_truncate: varargin to check
#                        decision: varargin to check
#                        cleaning + "".format(...) -> f"{...}"
#   1.6.0 (2024-07-01) - 
#                       mimo, SER_estimation, CPR_pilots, decision: same
#                        training schemes for cma/vae. Along with processing
#                        (1.3.0)
#   1.6.1 (2024-07-02) - CPR_pilots: adaptative filtering + Frame-FrameChannel
#                           for phase noise => Frame, along with processing
#                        (1.3.1) + correction
#                        receiver: cleaning
#   1.7.0 (2024-07-05) - [REMOVED] save_estimations, moved into rxdsp.mimo
#                        compensate_and_truncate:  t/r x['Symb_SER_real']
#                        definition moved to processing.init_processing
#                        along with processing (1.3.2)
#                        plot.decision: 'frame' argument added.
#                        along with general (1.5.3)
#                        remove_symbols: cleaning + adding "pilot" option
#                        decision: moving the symbol cutting to remove_symbols
#                        [NEW] front_end: for now its only power normalisation
#   1.7.1 (2024-07-06) - SER_estimation: plot title
#                        remove_symbols: plots to check
#                        find_shift, compensate_and_truncate: NSymbEq->NSymbSER
#                        pilot management: convolution splits the pilots on
#                           both current and previous batches.
#   1.7.2 (2024-07-07) - decision: normalising if PCS working for both CMA/VAE
#                           should be:  Xref = Xref/np.sqrt(np.mean(Ptx))
#                           + checking plot related to it
#                       - CPR_pilots: changing adaptive stop condition on diff
#                           diff -> rel_diff
#   1.7.3 (2024-07-10) - naming normalisation (*frame*-> *Frame*).
#                        along with main (1.4.3)
# ---------------------
#   2.0.0 (2024-07-12) - LIBRARY NAME CHANGED: LIB_GENERAL->LIB_PLOT + cleaning
# 
# ----- MAIN IDEA -----
#   Library for decision functions in (optical) telecommunications
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   [A1 ]Authors        : Fabio A. Barbosa, Sandro M. Rossin Darli A.A. Mello 
#   Title               : Phase and Frequency Recovery Algorithms for
#                         Probabilistically shaped transmission
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
import time

import lib_plot as plot
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
# compensate_and_truncate
# CPR_pilots                (1.5.0)
# decision
# find_shift
# front_end                 (1.7.0)
# mimo
# remove_symbols            (1.5.0)
# SNR_estimation
# =============================================================================


#%%
def receiver(tx,rx,saving):
            
    rx              = front_end(rx)
    rx, loss        = mimo(tx, rx, saving)#,'b4','after')
    rx              = remove_symbols(rx,'data')#,'data removal')
    rx              = CPR_pilots(tx, rx)#, 'demod','corr')#,'trace loss')#,'demod','corr')     # {align,demod,time trace pn, time trace pn loss, trace loss}
    rx              = SNR_estimation(tx, rx)
    rx              = decision(tx,rx)#,"const norm",'time decision')         # {time decision, const decision, const norm}
    tx,rx           = find_shift(tx, rx)#,'corr', 'err dec')                              # {corr, err dec}
    tx,rx           = compensate_and_truncate(tx,rx)#, 'err dec','corr')     # {corr, err dec}
    rx              = SER_estimation(tx, rx)#, 'err dec')                      # {err dec}

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
    shiftHI = np.argmax(xcorrHI)-rx["NSymbSER"]/2
    shiftHQ = np.argmax(xcorrHQ)-rx["NSymbSER"]/2
    shiftVI = np.argmax(xcorrVI)-rx["NSymbSER"]/2
    shiftVQ = np.argmax(xcorrVQ)-rx["NSymbSER"]/2
    
    assert shiftHI == shiftHQ == shiftVI == shiftVQ == 0,\
        "shift values should be equal to 0: {} - {} - {} - {}".format(\
          shiftHI,shiftHQ,shiftVI,shiftVQ)

    tx['Symb_SER_real'][0,rx['Frame']]  = THI[0]
    tx['Symb_SER_real'][1,rx['Frame']]  = THQ[0]
    tx['Symb_SER_real'][2,rx['Frame']]  = TVI[0]
    tx['Symb_SER_real'][3,rx['Frame']]  = TVQ[0]

    rx['Symb_SER_real'][0,rx['Frame']]  = RHI[0]
    rx['Symb_SER_real'][1,rx['Frame']]  = RHQ[0]
    rx['Symb_SER_real'][2,rx['Frame']]  = RVI[0]
    rx['Symb_SER_real'][3,rx['Frame']]  = RVQ[0]


    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and 'corr' in varargin:
        if rx['Frame']> rx['FrameChannel']:
            plot.xcorr_2x2(xcorrHI, xcorrHQ, xcorrVI, xcorrVQ,\
                       title =f"compensate_and_truncate - frame {rx['Frame']}",\
                       ref = 1, zoom = 1)


    if len(varargin) != 0 and 'err dec' in varargin:
        if rx["Frame"] >= rx["NFrames"]-1:
            t = tx['Symb_SER_real'][:,rx['Frame']]
            r = rx['Symb_SER_real'][:,rx['Frame']]
            plot.decisions(t, r, 3, rx['Frame'], NSymb = rx['NSymbBatch'],
                       title = f"compensate and truncate - frame {rx['Frame']}")
            # print(sum(r[0]==t[0])/len(r[0]))
            # print(sum(r[1]==t[1])/len(r[0]))
            # print(sum(r[2]==t[2])/len(r[0]))
            # print(sum(r[3]==t[3])/len(r[0]))
    # ---------------------------------------------------------------- to check

    del rx['tmp']

    return tx,rx


#%%
def CPR_pilots(tx,rx,*varargin):
    
    if tx['flag_phase_noise'] == 0 or rx['mode'].lower() == "blind":
        rx['sig_cpr_real'] = rx['sig_mimo_cut_real']
        return rx
    
    else:
# =============================================================================
#     maintenance
# =============================================================================

        what_pilots             = tx['pilots_info'][0]
        pilots_function         = what_pilots[0].lower()
        pilots_changes          = what_pilots[2].lower()
        tx_pilots               = tx['Symb_{}_cplx'.format(pilots_function)]
    

        if pilots_changes != "batchwise":
            tx_pilots_H_all     = mb.repmat(tx_pilots[0], (rx['NBatchFrame'],1))
            tx_pilots_V_all     = mb.repmat(tx_pilots[1], (rx['NBatchFrame'],1))
        else:          
            tx_pilots_H_all     = tx_pilots[0]
            tx_pilots_V_all     = tx_pilots[1]

        rx_HI       = rx["sig_mimo_cut_real"][0]
        rx_HQ       = rx["sig_mimo_cut_real"][1]
        rx_VI       = rx["sig_mimo_cut_real"][2]
        rx_VQ       = rx["sig_mimo_cut_real"][3]

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

        # shaping using convolution puts symboles at end of previous batch
        # and at beginning of current batch. the number of the latter is named
        # ``offset_conv"
        offset_conv         = rx['NSymb_pilots_cpr'] - tx['NSymbTaps']
        for k in range(int(rx['NBatchFrame_pilots'])):
            
            pilots_H_begin  = rx_H_rs[k, :offset_conv+1].reshape((1,-1))
            pilots_V_begin  = rx_V_rs[k, :offset_conv+1].reshape((1,-1))
            
            pilots_H_end    = rx_H_rs[k, -offset_conv+3:].reshape((1,-1))
            pilots_V_end    = rx_V_rs[k, -offset_conv+3:].reshape((1,-1))
            
            rx_H_pilots[k] = np.concatenate((pilots_H_begin,pilots_H_end),axis = 1)
            rx_V_pilots[k] = np.concatenate((pilots_V_begin,pilots_V_end),axis = 1)
        
        
    # ---------------------------------------------------------------- to check
        if len(varargin) != 0 and 'align' in varargin\
            and rx['Frame']> rx['FrameChannel']:
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
        del rx_HI, rx_HQ, rx_VI, rx_VQ
        del rx_H_rs, rx_V_rs


# =============================================================================
#     phase noise: estimation
# =============================================================================

        # conjugaison to remove modulation
        tmpH    = rx_H_pilots*np.conj(tx_pilots_H_roll[:,:rx['NSymb_pilots_cpr']])
        tmpV    = rx_V_pilots*np.conj(tx_pilots_V_roll[:,:rx['NSymb_pilots_cpr']])
        
        # normalisation [optional]
        tmpH   = tmpH/np.sqrt(power(tmpH))
        tmpV   = tmpV/np.sqrt(power(tmpV))
        
        # ------------------------------------------------------------ to check
        if len(varargin) != 0 and 'demod' in varargin\
            and rx['Frame']> rx['FrameChannel']:
            plot.constellations(sig1 = tmpH, sig2 = tmpV,\
                labels= ['H','V'], title = f"cpr_pilots, frame {rx['Frame']}")
        # ------------------------------------------------------------ to check
                
        # averaging over the pilots within the same batches
        phis_H_mean     = np.mean(np.angle(tmpH)*180/pi,axis = 1)             # [deg]
        phis_V_mean     = np.mean(np.angle(tmpV)*180/pi,axis = 1)             # [deg]

        del tmpH, tmpV, tx_pilots, rx_H_pilots, rx_V_pilots
        del tx_pilots_H, tx_pilots_V, tx_pilots_H_roll, tx_pilots_V_roll

        
        rx['PhaseNoise_pilots'][rx['Frame'],0]      = phis_H_mean
        rx['PhaseNoise_pilots'][rx['Frame'],1]      = phis_V_mean
     
# =============================================================================
#     adaptative noise filtering
# =============================================================================

        if tx['pn_filt_par']['adaptative'] == 1:
            
            rea             = 0        
            loss            = np.inf
            rel_diff_loss       = np.inf
                
            while (loss > tx['pn_filt_par']['err_tolerance'])\
                and (rea < tx['pn_filt_par']['niter_max'])\
                and (tx['pn_filt_par']['window_size'] < rx['NBatchFrame']/2)\
                and (abs(rel_diff_loss)>3e-3):
        
                pn_H_filter = maths.my_low_pass_filter(phis_H_mean, tx['pn_filt_par'])
                pn_V_filter = maths.my_low_pass_filter(phis_V_mean, tx['pn_filt_par'])
    
                loss_H      = maths.mse(pn_H_filter,tx['PhaseNoise_unique'][rx['Frame'],1:-1])
                loss_V      = maths.mse(pn_V_filter,tx['PhaseNoise_unique'][rx['Frame'],1:-1])
                loss        = np.round((loss_H+loss_V)/2,3)
                
                tx['pn_fil_losses'][rx['Frame'],rea] = loss
                tx['pn_filt_par']['window_size'] += 1
                
                if rea > 0:
                    rel_diff_loss = abs((tx['pn_fil_losses'][rx['Frame'],rea-1]-loss)/loss)
                    # if rx['Frame']>= rx['FrameChannel']:
                    #     print(rel_diff_loss)
                
                rea += 1
                
        # ------------------------------------------------------------ to check           
                if len(varargin) != 0 and "time trace pn loss" in varargin\
                    and rx['Frame'] >= rx['FrameChannel']:
                        batches     = np.linspace(1,rx['NBatchFrame_pilots'],rx['NBatchFrame_pilots'])
                        
                        plt.figure()
                        plt.plot(batches,tx['PhaseNoise_unique'][rx['Frame'],1:-1]*180/pi, label = 'ground truth')
                        plt.plot(batches, phis_H_mean, label = 'estimation raw')
                        plt.plot(batches,pn_H_filter, linestyle = 'dashed',label = "estimation filt")
                        plt.legend()
                        # plt.ylim(np.array([-90,90])/2)
                        plt.title(f"\n frame {rx['Frame']}, loss = {loss}, window = {tx['pn_filt_par']['window_size']}")
                        plt.show()
        # ------------------------------------------------------------ to check
            # end-while
            
        # ------------------------------------------------------------ to check
            if len(varargin) != 0 and "trace loss" in varargin\
                and rx['Frame'] >= rx['FrameChannel']:
                    batches     = np.linspace(1,rx['NBatchFrame_pilots'],rx['NBatchFrame_pilots'])
                    
                    plt.figure()
                    plt.plot(tx['pn_fil_losses'][rx['Frame']])
                    plt.xlim([0,1+rx['NBatchFrame']/2])
                    plt.title(f"\n frame {rx['Frame']}, loss = {loss}, window = {tx['pn_filt_par']['window_size']}")
                    plt.show()
        # ------------------------------------------------------------ to check
        
        
        else:
                        
            pn_H_filter = maths.my_low_pass_filter(phis_H_mean, tx['pn_filt_par'])
            pn_V_filter = maths.my_low_pass_filter(phis_V_mean, tx['pn_filt_par'])
            
        # ------------------------------------------------------------ to check
        if len(varargin) != 0 and "time trace pn" in varargin\
            and rx['Frame'] >= rx['FrameChannel']:
            batches     = np.linspace(1,rx['NBatchFrame_pilots'],rx['NBatchFrame_pilots'])
            
            plt.figure()
            plt.plot(batches,tx['PhaseNoise_unique'][rx['Frame'],1:-1]*180/pi, label = 'ground truth')
            plt.plot(batches,phis_H_mean, label = 'estimation raw')
            plt.plot(batches,pn_H_filter, linestyle = 'dashed',label = "estimation filt")
            plt.legend()
            # plt.ylim(np.array([-90,90])/2)
            plt.title(f"\n frame {rx['Frame']}, loss = {loss}, window = {tx['pn_filt_par']['window_size']}")
            plt.show()
        # ------------------------------------------------------------ to check
        
        # reseting fot the next frames
        tx['pn_filt_par']['window_size']        = 1

        rx['PhaseNoise_pilots'][rx['Frame'],0]  = pn_H_filter
        rx['PhaseNoise_pilots'][rx['Frame'],1]  = pn_V_filter
        
# =============================================================================
#     correction
# =============================================================================

        pn_H_filter     = np.repeat(pn_H_filter,rx['NSymbBatch'],axis = 0)*pi/180 # [rad]
        pn_V_filter     = np.repeat(pn_V_filter,rx['NSymbBatch'],axis = 0)*pi/180 # [rad]
        
        rx_H_corrected  = rx_H*np.exp(-1j*pn_H_filter)
        rx_V_corrected  = rx_V*np.exp(-1j*pn_V_filter)

        # ------------------------------------------------------------ to check
        if len(varargin) != 0 and 'corr' in varargin\
            and rx['Frame']> rx['FrameChannel']:
                rx_corrected    = np.concatenate((rx_H_corrected, rx_V_corrected),axis = 0)
                plot.constellations(sig1 = rx_corrected,sig2 = tx["sig_real"],\
                                        polar = 'H', labels = ['cpr','tx'],
                                        title = f"after cpr, frame {rx['Frame']}")
        # ------------------------------------------------------------ to check

        rx['sig_cpr_real']    = np.zeros((2*tx['Npolars'],rx_H_corrected.size))
        rx['sig_cpr_real'][0] = np.real(rx_H_corrected)
        rx['sig_cpr_real'][1] = np.imag(rx_H_corrected)
        rx['sig_cpr_real'][2] = np.real(rx_V_corrected)
        rx['sig_cpr_real'][3] = np.imag(rx_V_corrected)
        
        return rx


#%%
def decision(tx, rx, *varargin):           

    if tx['flag_phase_noise'] == 0:
        rx["sig_eq_real"] = rx["sig_mimo_cut_real"]
    else:
        rx["sig_eq_real"] = rx["sig_cpr_real"]
        
    if rx['mode'].lower() == 'blind':
        if rx['mimo'].lower() == "vae":
            ZHI     = np.round(np.reshape(rx["sig_eq_real"][0,rx['Frame']], (1, -1)).squeeze(), 4)
            ZHQ     = np.round(np.reshape(rx["sig_eq_real"][1,rx['Frame']], (1, -1)).squeeze(), 4)
            ZVI     = np.round(np.reshape(rx["sig_eq_real"][2,rx['Frame']], (1, -1)).squeeze(), 4)
            ZVQ     = np.round(np.reshape(rx["sig_eq_real"][3,rx['Frame']], (1, -1)).squeeze(), 4)
            Prx     = power(rx["sig_eq_real"][:,rx['Frame']],flag_real2cplx=1,flag_flatten=1)
        else:
            ZHI     = np.round(np.reshape(rx["sig_eq_real"][0], (1, -1)).squeeze(), 4)
            ZHQ     = np.round(np.reshape(rx["sig_eq_real"][1], (1, -1)).squeeze(), 4)
            ZVI     = np.round(np.reshape(rx["sig_eq_real"][2], (1, -1)).squeeze(), 4)
            ZVQ     = np.round(np.reshape(rx["sig_eq_real"][3], (1, -1)).squeeze(), 4)

            Prx     = power(rx["sig_eq_real"],flag_real2cplx=1,flag_flatten=1)
    
    else:
        if len(varargin) != 0 and "pilots removal" in varargin:
            Zcut    = remove_symbols(rx,tx, 'pilots removal')
        else:
            Zcut    = remove_symbols(rx,tx)
            
        ZHI     = np.reshape(Zcut[0],(1,-1))
        ZHQ     = np.reshape(Zcut[1],(1,-1))
        ZVI     = np.reshape(Zcut[2],(1,-1))
        ZVQ     = np.reshape(Zcut[3],(1,-1))
        
        Prx     = [0,0]
        Prx[0]  = power((ZHI+1j*ZHQ).flatten())
        Prx[1]  = power((ZVI+1j*ZVQ).flatten())
        
    
    ZHInorm     = (ZHI/np.sqrt(Prx[0])).reshape((1,-1))
    ZHQnorm     = (ZHQ/np.sqrt(Prx[0])).reshape((1,-1))
    ZVInorm     = (ZVI/np.sqrt(Prx[1])).reshape((1,-1))
    ZVQnorm     = (ZVQ/np.sqrt(Prx[1])).reshape((1,-1))
    

    # ---------------------------------------------------------------- to check
    # checking normalisation
    ZHnorm  = np.array([ZHInorm+1j*ZHQnorm,ZVInorm+1j*ZVQnorm])
    assert abs(sum(power(ZHnorm))-2) <= 1e-1,\
        f'power should be close to 1 per polar, got {power(ZHnorm)}'
    del ZHnorm
    # ---------------------------------------------------------------- to check

    M           = int(tx['mod'][0:-3])                  # 'M' for M-QAM
    ZHI_ext     = np.tile(ZHInorm, [int(np.sqrt(M)), 1])
    ZHQ_ext     = np.tile(ZHQnorm, [int(np.sqrt(M)), 1])
    ZVI_ext     = np.tile(ZVInorm, [int(np.sqrt(M)), 1])
    ZVQ_ext     = np.tile(ZVQnorm, [int(np.sqrt(M)), 1])
        
    X_alphabet  = tx['const_affixes']
    Px_alphabet = power(X_alphabet)
    Xref        = X_alphabet/np.sqrt(Px_alphabet)
    Ptx         = power(np.reshape(tx["Symb_real"],(4,-1)), flag_real2cplx=1)
    
    if tx['nu'] != 0:
        Xref    = Xref/np.sqrt(np.mean(Ptx))

    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and "const norm" in varargin\
        and rx['Frame']> rx['FrameChannel']:
            ZXnorm      = np.array([ZHInorm+1j*ZHQnorm,ZVInorm+1j*ZVQnorm]).squeeze()
            Xref_check  = Xref.reshape((1,-1))
        
            plot.constellations(ZXnorm,Xref_check, labels =['eq norm',"ref"],\
                    sps = 1, title = f"rxdsp.decision - frame {rx['Frame']}")
    # ---------------------------------------------------------------- to check

    Ref     = np.tile(np.unique(np.real(Xref)), [rx['NSymbSER'],1]).transpose()
    
    Err_HI  = abs(ZHI_ext - Ref).astype(np.float16)
    Err_HQ  = abs(ZHQ_ext - Ref).astype(np.float16)
    Err_VI  = abs(ZVI_ext - Ref).astype(np.float16)
    Err_VQ  = abs(ZVQ_ext - Ref).astype(np.float16)
    
    
    del ZHI_ext, ZHQ_ext, ZVI_ext, ZVQ_ext

    if tx["nu"] != 0: # probabilistic shaping
    
        if rx['mimo'].lower() != "vae" and type(tx["prob_amps"]) != torch.Tensor:
            tx["prob_amps"]         = torch.tensor(tx["prob_amps"])

        prob_amps   = np.tile(tx['prob_amps'].unsqueeze(1), [1, rx['NSymbSER']])

        Err_HI      = Err_HI ** 2 - 1/rx['SNR']*np.log(prob_amps)
        Err_HQ      = Err_HQ ** 2 - 1/rx['SNR']*np.log(prob_amps)
        Err_VI      = Err_VI ** 2 - 1/rx['SNR']*np.log(prob_amps)
        Err_VQ      = Err_VQ ** 2 - 1/rx['SNR']*np.log(prob_amps)

    tmp_HI  = np.argmin(Err_HI, axis=0)
    tmp_HQ  = np.argmin(Err_HQ, axis=0)
    tmp_VI  = np.argmin(Err_VI, axis=0)
    tmp_VQ  = np.argmin(Err_VQ, axis=0)
    
    ZHI_dec = tx['amps'][tmp_HI].reshape((1,-1))
    ZHQ_dec = tx['amps'][tmp_HQ].reshape((1,-1))
    ZVI_dec = tx['amps'][tmp_VI].reshape((1,-1))
    ZVQ_dec = tx['amps'][tmp_VQ].reshape((1,-1))

    del Err_HI, Err_HQ, Err_VI, Err_VQ
    del tmp_HI, tmp_HQ, tmp_VI, tmp_VQ
    
    rx['Symb_real_dec'][0, rx['Frame']] = ZHI_dec
    rx['Symb_real_dec'][1, rx['Frame']] = ZHQ_dec
    rx['Symb_real_dec'][2, rx['Frame']] = ZVI_dec
    rx['Symb_real_dec'][3, rx['Frame']] = ZVQ_dec

    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and "time decision" in varargin:
        if rx["Frame"] >= rx["FrameChannel"]:
            t = [ZHInorm,ZHQnorm,ZVInorm,ZVQnorm]
            r = [ZHI_dec,ZHQ_dec,ZVI_dec,ZVQ_dec]
            plot.decisions(t, r, 5, rx['Frame'], NSymb = rx['NSymbBatch'],
                               title = f"rxdsp.decision - frame {rx['Frame']}")

    if len(varargin) != 0 and "const decision" in varargin:
        if rx["Frame"] >= rx["FrameChannel"]:
            ZXnorm = np.array([ZHInorm+1j*ZHQnorm,ZVInorm+1j*ZVQnorm]).squeeze()
            # TXclean = np.reshape(tx["Symb_real"],(4,-1))
            TX = np.reshape(tx["sig_real"],(4,-1))
            plot.constellations(TX,ZXnorm, labels =['tx',"eq"],\
                                title = f"rxdsp.decision - frame {rx['Frame']}")
    # ---------------------------------------------------------------- to check

    return rx


#%%
def find_shift(tx,rx,*varargin):

    # maintenance
    ref     = tx['Symb_real'].numpy()
    
    if rx['mode'].lower() != 'blind':
        
        offset_conv = rx['NSymb_pilots_cpr'] - tx['NSymbTaps']        
        tmp         = ref[:,:,rx['NSymbBatch']:-rx['NSymbBatch']]
        tmp         = np.reshape(tmp,(2,2,rx['NBatchFrame_pilots'],-1))
        tmp         = tmp[:,:,:,offset_conv+1:-offset_conv+3]
        tmp         = np.reshape(tmp,(2,2,-1))
        del ref
        ref         = tmp
    
    sig     = rx['Symb_real_dec']
    
    if ref.shape[-1] != sig.shape[-1]:
        Nsb_to_remove   = int(rx['NSymbCut_tot']-1)
        ref             = ref[:,:,Nsb_to_remove:]
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
    shiftHI = np.argmax(xcorrHI)-rx["NSymbSER"]/2
    shiftHQ = np.argmax(xcorrHQ)-rx["NSymbSER"]/2
    shiftVI = np.argmax(xcorrVI)-rx["NSymbSER"]/2
    shiftVQ = np.argmax(xcorrVQ)-rx["NSymbSER"]/2

    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and 'corr' in varargin\
        and rx['Frame'] >= rx['FrameChannel']:
        plot.xcorr_2x2(xcorrHI, xcorrHQ, xcorrVI, xcorrVQ,\
              title = f"rxdsp.find_shift - frame {rx['Frame']}", ref=1, zoom=1)


    if len(varargin) != 0 and 'err dec' in varargin\
        and rx['Frame'] >= rx['FrameChannel']:
        Tx  = np.concatenate((THI, THQ, TVI, TVQ),axis = 0)
        Dec = np.concatenate((RHI, RHQ, RVI, RVQ),axis = 0)
        plot.decisions(Tx,Dec,5, rx['Frame'], rx['NSymbBatch'],\
                       title = f"rxdsp.find_shift - frame {rx['Frame']}")
    # ---------------------------------------------------------------- to check

    assert shiftHI == shiftHQ == shiftVI == shiftVQ,\
        "shift values should be equal :"+\
           f"{shiftHI} - {shiftHQ} - {shiftVI} - {shiftVQ}"
           
    rx['NSymbShift']    = int(shiftHI)
    

    return tx,rx

#%%
def front_end(rx):
    
    if rx['norm_power'] == 1:
        rx['sig_real'] = maths.normalise_power(rx['sig_real'])
            
    return rx

#%%
def mimo(tx,rx,saving,*varargin):

    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and 'b4' in varargin:
        if rx['Frame']> rx['FrameChannel']:
            plot.constellations(sig1 = rx['sig_real'],title =
                                              f"rx b4 mimo {rx['Frame']}")
    # ---------------------------------------------------------------- to check
    
    
    if rx['mimo'].lower() == "vae":
        with torch.set_grad_enabled(True):
            for BatchNo in range(rx["NBatchFrame"]):

                rx              = kit.train_vae(BatchNo,rx,tx)
                rx,loss         = kit.compute_vae_loss(tx,rx)
                maths.update_fir(loss,rx['optimiser'])

                
                if rx['save_channel_batch']:
                    # rx['h_est_batch'].append(rx['h_est'].detach().numpy())
                    rx['h_est_batch'].append(rx['h_est'].tolist())
                    # print(len(rx['h_est_batch']))
                    

    
    # ---------------------------------------------------------------- to check
        if len(varargin) != 0 and "loss" in varargin:
            plot.loss_batch(rx,saving,['kind','law',"std",'linewidth'],"Llikelihood")
            plot.loss_batch(rx,saving,['kind','law',"std",'linewidth'],"DKL")
            plot.loss_batch(rx,saving,['kind','law',"std",'linewidth'],"losses")
    # ---------------------------------------------------------------- to check

    elif rx['mimo'].lower() == "cma" :
        rx,loss = kit.CMA(tx,rx)



    # ---------------------------------------------------------------- to check
        if len(varargin) != 0 and "loss" in varargin:
            plot.loss_cma(rx,saving,['kind','law',"std",'linewidth'],"x")
    # ---------------------------------------------------------------- to check
                

    else:
        loss = []
        

    # ---------------------------------------------------------------- to check        
    if len(varargin) != 0 and "after" in varargin:
        if rx['Frame']> rx['FrameChannel']:
            if rx['mimo'].lower() == "cma" :
                plot.constellations(rx["sig_mimo_real"],\
                                    title =f"rx after mimo {rx['Frame']}")
            else:
                plot.constellations(rx["sig_mimo_real"][:,rx['Frame']],\
                                    title =f"rx after mimo {rx['Frame']}")            

    if len(varargin) != 0 and "fir" in varargin:
        if rx['Frame']> rx['FrameChannel']:
            plot.fir(rx, title =f"fir after mimo {rx['Frame']}")
    # ---------------------------------------------------------------- to check
    
    # louis, do not change, for now, .tolist() into .detach().numpy()
    #  as it fails when processing with matlab
    rx["h_est_frame"].append(rx["h_est"].tolist())
    
    return rx,loss
    
    

#%%
def remove_symbols(rx,what, *varargin):

    if type(what) == str:
        if rx['mimo'].lower() != 'vae':
            if rx['mode'].lower() == 'blind':
                rx['NSymbFrame_b4cpr']= rx['NSymbFrame']-rx['NSymbCut_tot']+1
                tmp = rx["sig_mimo_real"][:,rx['NSymbCut']:-rx['NSymbCut']-1]
            
            else:
        
                rx['NSymbFrame_b4cpr']= rx['NSymbFrame']-2*rx['NSymbBatch']
                tmp = rx["sig_mimo_real"][:,rx['NSymbBatch']:-rx['NSymbBatch']-1]
                    
            rx["sig_mimo_cut_real"] = tmp
    
        else:
            rx["sig_mimo_cut_real"] = rx["sig_mimo_real"]

        # ---------------------------------------------------------------- to check
        if len(varargin) != 0 and 'data removal' in varargin:
            if rx['Frame']> rx['FrameChannel']:
                plot.constellations(rx["sig_mimo_cut_real"],
                                        title = f"remove data - f {rx['Frame']}")
        # ---------------------------------------------------------------- to check
        return rx
    
    if type(what) == dict:
        offset_conv     = rx['NSymb_pilots_cpr'] - what['NSymbTaps']
        
        ZHItmp  = np.round(np.reshape(rx["sig_eq_real"][0], (1, -1)).squeeze(), 4)
        ZHQtmp  = np.round(np.reshape(rx["sig_eq_real"][1], (1, -1)).squeeze(), 4)
        ZVItmp  = np.round(np.reshape(rx["sig_eq_real"][2], (1, -1)).squeeze(), 4)
        ZVQtmp  = np.round(np.reshape(rx["sig_eq_real"][3], (1, -1)).squeeze(), 4)

        ZHIrs   = np.reshape(ZHItmp,(rx['NBatchFrame_pilots'],-1))
        ZHQrs   = np.reshape(ZHQtmp,(rx['NBatchFrame_pilots'],-1))
        ZVIrs   = np.reshape(ZVItmp,(rx['NBatchFrame_pilots'],-1))
        ZVQrs   = np.reshape(ZVQtmp,(rx['NBatchFrame_pilots'],-1))
        
        ZHIcut  = ZHIrs[:,offset_conv+1:-offset_conv+3]
        ZHQcut  = ZHQrs[:,offset_conv+1:-offset_conv+3]
        ZVIcut  = ZVIrs[:,offset_conv+1:-offset_conv+3]
        ZVQcut  = ZVQrs[:,offset_conv+1:-offset_conv+3]

        Zcut    = [ZHIcut,ZHQcut,ZVIcut,ZVQcut]
        
        del ZHItmp,ZHQtmp,ZVItmp,ZVQtmp
        
    # ---------------------------------------------------------------- to check
        if len(varargin) != 0 and 'pilots removal' in varargin:
            if rx['Frame']> rx['FrameChannel']:            
                plt.figure()
                #
                plt.subplot(2,2,1)
                plt.plot(ZHIrs[:5,:15].T)
                plt.title("begin batches - b4 cut")
                #
                plt.subplot(2,2,2)
                plt.plot(ZHIrs[:5,-15:].T)
                plt.title("end batches - b4 cut")
                #
                plt.subplot(2,2,3)
                plt.plot(ZHIcut[:5,:15].T)
                plt.title("begin batches - after cut")
                #
                plt.subplot(2,2,4)
                plt.plot(ZHIcut[:5,-15:].T)
                plt.title("end batches - after cut")
                #
                plt.show()
    # ---------------------------------------------------------------- to check
    
        return Zcut
    



#%%
def SER_estimation(tx,rx,*varargin):
    
    RH      = rx['Symb_SER_real'][0,rx['Frame']]+1j*rx['Symb_SER_real'][1,rx['Frame']]
    RV      = rx['Symb_SER_real'][2,rx['Frame']]+1j*rx['Symb_SER_real'][3,rx['Frame']]

    TH      = tx['Symb_SER_real'][0,rx['Frame']]+1j*tx['Symb_SER_real'][1,rx['Frame']]
    TV      = tx['Symb_SER_real'][2,rx['Frame']]+1j*tx['Symb_SER_real'][3,rx['Frame']]


    rx["SER_valid"][0,rx['Frame']]  = sum(RH!=TH)/rx['NSymbSER']
    rx["SER_valid"][1,rx['Frame']]  = sum(RV!=TV)/rx['NSymbSER']

    # ---------------------------------------------------------------- to check
    if len(varargin) != 0 and 'err dec' in varargin:
        if rx['Frame']> rx['FrameChannel']:
            tmpErrH = tx['Symb_SER_real'][0,rx['Frame']] != rx['Symb_SER_real'][0,rx['Frame']]
            tmpErrV = tx['Symb_SER_real'][1,rx['Frame']] != rx['Symb_SER_real'][1,rx['Frame']]
            
            # for k in range(10):
            #     plt.figure()
            #     a = k*(rx['NSymbBatch'])
            #     b = a+rx['NSymbBatch']
            #     plt.plot(tmpErrH[a:b], label = 'H')
            #     plt.plot(tmpErrV[a:b], label = 'V', linestyle = "dashed")
            #     plt.legend()
            #     plt.title(f"ser_estimation - f{rx['Frame']}, [{a},{b}]")
            #     plt.show()
                
    
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(tmpErrH, label = 'H')
            plt.title(f"ser_estimation H - f{rx['Frame']}")
            plt.subplot(2,1,2)
            plt.plot(tmpErrV, label = 'V')
            plt.title(f"ser_estimation V - f{rx['Frame']}")
            plt.show()
    # ---------------------------------------------------------------- to check
    
    return rx

#%% [C1]
def SNR_estimation(tx,rx):
    
    if rx["mimo"].lower() == "vae":
        rx["SNRdB_est"][rx['Frame']]      = tx["const_pow_mean"]/torch.mean(rx['Pnoise_batches'])
        rx['Pnoise_est'][:,rx['Frame']]   = torch.mean(rx['Pnoise_batches'],dim=1)  

    return rx


