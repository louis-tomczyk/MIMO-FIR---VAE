# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.5.0
#   Date            : 2024-06-20
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
#   1.4.1 (2024-06-06) - find_shift: NSymb_to_remove changed (no use of any weird "reference" number
#                           of taps). Ntaps -> NsampTaps
#   1.5.0 (2024-06-20) - [NEW] CPR_pilots: for MQAM modulations, Carrier Phase
#                           Recovery is QPSK pilot aided. see [A1]
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
#   [C2] Authors        : Jingtian Liu, Élie Awwad, Louis Tomczyk
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

import lib_kit as kit
import lib_matlab as mb
import lib_maths as maths
import lib_general as gen
import lib_rxdsp as rxdsp

from lib_matlab import clc
from lib_misc import KEYS as keys
from lib_maths import power


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
        
        # loss            = np.zeros((rx['NSymbFrame']+1,tx['Npolars']))

        # rx['sig_eq_real_cma']       = np.zeros((4,rx['NSymbFrame']+1))
        # rx['sig_eq_real_cma'][0]    = rx['sig_real'][0,0:-2:tx['Nsps']]
        # rx['sig_eq_real_cma'][1]    = rx['sig_real'][1,0:-2:tx['Nsps']]
        # rx['sig_eq_real_cma'][2]    = rx['sig_real'][2,0:-2:tx['Nsps']]
        # rx['sig_eq_real_cma'][3]    = rx['sig_real'][3,0:-2:tx['Nsps']]
        
        # print(rx['Frame'])

        rx              = remove_symbols(rx,'data')
        rx              = CPR_pilots(tx, rx)
        # rx              = save_estimations(rx)
        # rx              = SNR_estimation(tx, rx)
        # rx              = SER_estimation(tx, rx)
        

        # if rx['Frame'] >= rx['FrameChannel']:
        #     gen.plot_constellations(sig1 = tx['sig_real'],title ='tx')
        #     gen.plot_constellations(sig1 = rx['sig_real'],title ='rx b4 mimo')
        #     gen.plot_constellations(sig1 = rx['sig_eq_real_cma'],title ='rx after mimo')        
        
        # ------------------------------------------------------------------- #
        # k           = 0
        
        # ref_HI      = np.real(tx['Symb_{}_cplx'.format(tx['pilots_info'][0][0])][0])
        # ref_HQ      = np.imag(tx['Symb_{}_cplx'.format(tx['pilots_info'][0][0])][0])
        # ref_VI      = np.real(tx['Symb_{}_cplx'.format(tx['pilots_info'][0][0])][1])
        # ref_VQ      = np.imag(tx['Symb_{}_cplx'.format(tx['pilots_info'][0][0])][1])
        
        # TX_HI       = tx['sig_real'][0,k*320+0:k*320+10]
        # TX_HQ       = tx['sig_real'][1,k*320+0:k*320+10]
        # TX_VI       = tx['sig_real'][2,k*320+0:k*320+10]
        # TX_VQ       = tx['sig_real'][3,k*320+0:k*320+10]
            
        # RX_HI       = rx['sig_real'][0,k*320+0:k*320+10].numpy()
        # RX_HQ       = rx['sig_real'][1,k*320+0:k*320+10].numpy()
        # RX_VI       = rx['sig_real'][2,k*320+0:k*320+10].numpy()
        # RX_VQ       = rx['sig_real'][3,k*320+0:k*320+10].numpy()
        
        # RX_ASE_H    = rx['Noise'][0,k*320+0:k*320+10]
        # RX_ASE_V    = rx['Noise'][1,k*320+0:k*320+10]
            
        # ref_H       = ref_HI+1j*ref_HQ
        # ref_V       = ref_VI+1j*ref_VQ
        
        # TX_H        = TX_HI+1j*TX_HQ
        # TX_V        = TX_VI+1j*TX_VQ
        
        # RX_H        = RX_HI+1j*RX_HQ
        # RX_V        = RX_VI+1j*RX_VQ
            
        # amp_TX_H    = np.abs(TX_H)
        # amp_TX_V    = np.abs(TX_V)
        # amp_RX_H    = np.abs(RX_H)
        # amp_RX_V    = np.abs(RX_V)
        # amp_RX_ase  = np.abs(RX_ASE_H)
        # amp_RX_ase  = np.abs(RX_ASE_H)
        
        # phi_TX_H    = np.angle(TX_H)
        # phi_TX_V    = np.angle(TX_V)
        # phi_RX_H    = np.angle(RX_H)
        # phi_RX_V    = np.angle(RX_V)
        
        # diff_amp_H  = np.abs(amp_RX_H-amp_TX_H)
        # diff_amp_V  = np.abs(amp_RX_V-amp_TX_V)

        # diff_phi_H  = np.abs(phi_RX_H-phi_TX_H)
        # diff_phi_V  = np.abs(phi_RX_V-phi_TX_V)
        # ------------------------------------------------------------------- #
        
        return rx, loss


    
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
def CPR_pilots(tx,rx):
    
# =============================================================================
#     maintenance
# =============================================================================

    if rx['mode'].lower() == 'blind'\
    or "sig_eq_real_cma" not in rx\
    or rx['Frame'] < rx['FrameChannel']:
        return rx

    else:
        what_pilots             = tx['pilots_info'][0]
        pilots_function         = what_pilots[0].lower()
        pilots_changes          = what_pilots[2].lower()
        pilots                  = tx['Symb_{}_cplx'.format(pilots_function)]

# =============================================================================
#     stuffing
# =============================================================================
    
        if pilots_changes != "batchwise":
            pilots_H_all        = mb.repmat(pilots, (rx['NBatchFrame'],1))
            pilots_V_all        = mb.repmat(pilots, (rx['NBatchFrame'],1))
        else:
            pilots_H_all        = pilots[0]
            pilots_V_all        = pilots[1]
    
        pilots_H                = pilots_H_all[:,0:rx['NSymbPilots_cma']]
        pilots_V                = pilots_V_all[:,0:rx['NSymbPilots_cma']]
       
        [H_stuffed, V_stuffed]  = maths.zero_stuffing(
                                             sig_in = [pilots_H,pilots_V],\
                                             Nzeros = rx['Nzeros_stuffing'],
                                             Ntimes = rx['NBatchFrame'])

        rx['pilots_{}_cplx_stuffed'.format(pilots_function)] =\
                 np.concatenate((H_stuffed,V_stuffed),axis = 0)

        rx                      = remove_symbols(rx,'pilots_cpr')
        
        del pilots_H, pilots_V, H_stuffed, V_stuffed
        del pilots, pilots_H_all, pilots_V_all
        
# =============================================================================
#     logistics
# =============================================================================

        rx_HI    = rx['sig_eq_real'][0]
        rx_HQ    = rx['sig_eq_real'][1]
        rx_VI    = rx['sig_eq_real'][2]
        rx_VQ    = rx['sig_eq_real'][3]

        rx_H     = np.expand_dims(rx_HI+1j*rx_HQ,axis = 0)
        rx_V     = np.expand_dims(rx_VI+1j*rx_VQ,axis = 0)


        pilots_H_stuffed    = np.expand_dims(
            rx['pilots_{}_cplx_stuffed'.format(pilots_function)][0],axis = 0)
        pilots_V_stuffed    = np.expand_dims(
            rx['pilots_{}_cplx_stuffed'.format(pilots_function)][1],axis = 0)


        gen.plot_constellations(sig1 = rx_H, sig2 = pilots_H_stuffed,
            polar = 'H',labels=['RX','pilots'],title = "received and pilots")
        
        del rx_HI, rx_HQ, rx_VI, rx_VQ
        
# =============================================================================
#       conjugaison
# =============================================================================
        tmpH    = rx_H*np.conj(pilots_H_stuffed)
        tmpV    = rx_V*np.conj(pilots_V_stuffed)
    

        tmpH    = np.expand_dims(tmpH[tmpH!=0], axis = 0)
        tmpV    = np.expand_dims(tmpV[tmpV!=0], axis = 0)
        
        # the supplementary rotation of pi/4 is to prevent the ambiguity
        # in angle estimation
        tmpHn   = tmpH/np.abs(tmpH)*np.exp(1j*pi/4)
        tmpVn   = tmpV/np.abs(tmpV)*np.exp(1j*pi/4)

        del rx_H, rx_V, tmpH, tmpV
        gen.plot_constellations(tmpHn,polar = 'both',
                    title = 'tmp {} = z_n*a_n^*'.format(rx['Frame']))
        

# =============================================================================
#     decision
# =============================================================================
        
        phis_H          = np.angle(tmpHn)
        phis_V          = np.angle(tmpVn)

        phis_H_batches  = np.reshape(phis_H,(-1,rx['NSymbPilots_cma']))
        phis_V_batches  = np.reshape(phis_V,(-1,rx['NSymbPilots_cma']))
    
        ref_angles      = np.array([[-3,-1,+1,+3]]).T*pi/4
        ref_angles_rep  = mb.repmat(ref_angles, (1,rx['NSymbPilots_cma']))
        
        argmins_H       = np.zeros((rx['NBatchFrame_pilots'],
                                    rx['NSymbPilots_cma'])).astype(int)
        argmins_V       = np.zeros((rx['NBatchFrame_pilots'],
                                    rx['NSymbPilots_cma'])).astype(int)
        
        phis_dec_H      = np.zeros((rx['NBatchFrame_pilots'],
                                    rx['NSymbPilots_cma']))
        phis_dec_V      = np.zeros((rx['NBatchFrame_pilots'],
                                    rx['NSymbPilots_cma']))
        
        
        for k in range(rx['NBatchFrame_pilots']):
            phis_H_batch_k  = mb.repmat(phis_H_batches[k],(4,1))
            phis_V_batch_k  = mb.repmat(phis_V_batches[k],(4,1))
            
            diff_H          = np.abs(phis_H_batch_k - ref_angles_rep)
            diff_V          = np.abs(phis_V_batch_k - ref_angles_rep)
            
            argmins_H[k]    = np.argmin(diff_H,axis = 0)
            argmins_V[k]    = np.argmin(diff_V,axis = 0)
            
            
        for k in range(rx['NBatchFrame_pilots']):
            for j in range(rx['NSymbPilots_cma']):
                phis_dec_H[k,j] = ref_angles[argmins_H[k,j]][0]
                phis_dec_V[k,j] = ref_angles[argmins_V[k,j]][0]

        # if wanna check
        # plt.figure()
        # #
        # plt.subplot(1,2,1)
        # plt.plot(phis_H.T,label = 'rx')
        # plt.plot(phis_dec_H.flatten(),label = 'dec')
        # plt.legend()
        # plt.title("pol H")
        # #
        # plt.subplot(1,2,2)
        # plt.plot(phis_V.T,label = 'rx')
        # plt.plot(phis_dec_V.flatten(),label = 'dec')
        # plt.legend()
        # plt.title("pol V")
        # #
        # plt.show()

        del tmpHn, tmpVn, phis_H, phis_V, ref_angles, ref_angles_rep
        del phis_H_batch_k, phis_V_batch_k, diff_H, diff_V
        del argmins_H, argmins_V
        
# =============================================================================
#     phase noise
# =============================================================================

        for k in range(rx['NBatchFrame_pilots']):

            rx['PhaseNoise_pilots'][rx['Frame']-rx['FrameChannel'],0,k] =\
                                    phis_H_batches[k]-phis_dec_H[k]
                                    
            rx['PhaseNoise_pilots'][rx['Frame']-rx['FrameChannel'],1,k] =\
                                        phis_V_batches[k]-phis_dec_V[k]
            
        return rx




#%%
def find_pilots(tx,rx):
    
    if rx['mimo'].lower() != 'blind'\
    and rx['mimo'].lower() == 'cma'\
    and rx['Frame'] >= rx['FrameChannel']:
        for k in range(len(tx['pilots_info'])):
            
            # maintenance
            ref     = tx["pilots_{}_real".format(tx['pilots_info'][k][0])]
            sig     = rx['sig_eq_real'].squeeze()
            
            # sig     = rx['sig_real']
            
            THI     = ref[0]
            THQ     = ref[1]
            TVI     = ref[2]
            TVQ     = ref[3]
        
            RHI     = np.reshape(sig[0],(1,-1)).squeeze()
            RHQ     = np.reshape(sig[1],(1,-1)).squeeze()
            RVI     = np.reshape(sig[2],(1,-1)).squeeze()
            RVQ     = np.reshape(sig[3],(1,-1)).squeeze()
        
            # correlation between each channel
            xcorrHI = np.correlate(THI,RHI,mode = 'full')
            xcorrHQ = np.correlate(THQ,RHQ,mode = 'full')
            xcorrVI = np.correlate(TVI,RVI,mode = 'full')
            xcorrVQ = np.correlate(TVQ,RVQ,mode = 'full')
        
            # getting the shift
            shiftHI = np.argmax(xcorrHI)
            shiftHQ = np.argmax(xcorrHQ)
            shiftVI = np.argmax(xcorrVI)
            shiftVQ = np.argmax(xcorrVQ)
            
            # thresh = 6
            # print(np.sum(xcorrHI>thresh))
            # print(np.sum(xcorrHQ>thresh))
            # print(np.sum(xcorrVI>thresh))
            # print(np.sum(xcorrVQ>thresh))
        
            # displaying the correaltions to check
            gen.plot_xcorr_2x2(xcorrHI, xcorrHQ, xcorrVI, xcorrVQ, "test {} - find pilots".format(rx['Frame']))
            
            
            
            # for BatchNo in range(int(rx['NBatchFrame']/2)):
            #     NBatch  = 2
            #     a       = int(rx['NSymbFrame']/rx['NBatchFrame']+1)*BatchNo
            #     b       = int(a+rx['NSymbFrame']/rx['NBatchFrame']*NBatch)
            #     plt.plot(xcorrHI[a:b])
            #     distances = np.zeros(1)
            #     plt.title('{}'.format(rx['NSymbBatch']-np.argmax(xcorrHI[a:b])))
            #     plt.show()
        
        
            rx['PilotsShift']    = np.array([shiftHI,shiftHQ,shiftVI,shiftVQ]).astype(int)
    
    return rx


#%%
def find_shift(tx,rx):

    # maintenance
    ref     = tx['Symb_real'].numpy()
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
    # gen.plot_xcorr_2x2(xcorrHI, xcorrHQ, xcorrVI, xcorrVQ, "test - find shift", ref=1, zoom=1)

    assert shiftHI == shiftHQ == shiftVI == shiftVQ, "shift values should be equal : {} - {} - {} - {}".format(shiftHI,shiftHQ,shiftVI,shiftVQ)
    rx['NSymbShift']    = int(shiftHI)


    return tx,rx


#%% [C1]
def mimo(tx,rx,saving):

    if rx['mimo'].lower() == "vae":
        with torch.set_grad_enabled(True):
            for BatchNo in range(rx["NBatchFrame"]):

                rx              = kit.train_self(BatchNo,rx,tx)
                rx,loss         = kit.compute_vae_loss(tx,rx)
                maths.update_fir(loss,rx['optimiser'])

        # gen.plot_constellations(rx['sig_eq_real'][:,rx['Frame'],:,:], title = "RX f-{}".format(rx['Frame']))
            
            # gen.plot_loss_batch(rx,saving,['kind','law',"std",'linewidth'],"Llikelihood")
            # plot_loss_batch(rx,saving,['kind','law',"std",'linewidth'],"DKL")
            # plot_loss_batch(rx,saving,['kind','law',"std",'linewidth'],"losses")

        return rx,loss

    elif rx['mimo'].lower() == "cma" :

        if rx["Frame"]>rx['FrameChannel']-1:
            rx,loss = kit.CMA(tx,rx)
            # gen.plot_loss_cma(rx,saving,['kind','law',"std",'linewidth'],"x")
            # gen.plot_constellations(rx['sig_eq_real'],polar='both', title = "RX f-{}".format(rx['Frame']))
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
def remove_symbols(rx,what):

    if rx['Frame'] >= rx['FrameChannel']:
        
        if what.lower() == 'data':
            if rx['mode'].lower() == 'blind':
                rx['sig_eq_real'][0] = rx['sig_eq_real_cma'][0,rx['NSymbCut']:-rx['NSymbCut']-1]
                rx['sig_eq_real'][1] = rx['sig_eq_real_cma'][1,rx['NSymbCut']:-rx['NSymbCut']-1]
                rx['sig_eq_real'][2] = rx['sig_eq_real_cma'][2,rx['NSymbCut']:-rx['NSymbCut']-1]
                rx['sig_eq_real'][3] = rx['sig_eq_real_cma'][3,rx['NSymbCut']:-rx['NSymbCut']-1]
            
            else:
        
                NSymbTot    = rx['NSymbFrame']-2*rx['NSymbBatch']
                tmp         =  np.zeros((4,NSymbTot))

                tmp = rx['sig_eq_real_cma'][:,rx['NSymbBatch']:-rx['NSymbBatch']-1]
                    
                del rx['sig_eq_real']
                rx['sig_eq_real'] = tmp
                # gen.plot_constellations(rx['sig_eq_real'],polar = 'both')
        else:
            tmpH    = rx['{}_cplx_stuffed'.format(what)][0][rx['NSymbBatch']:-rx['NSymbBatch']]
            tmpV    = rx['{}_cplx_stuffed'.format(what)][1][rx['NSymbBatch']:-rx['NSymbBatch']]
            
            tmpH    = np.expand_dims(tmpH, axis= 0)
            tmpV    = np.expand_dims(tmpV, axis= 0)
            del rx['{}_cplx_stuffed'.format(what)]
            rx['{}_cplx_stuffed'.format(what)] = np.concatenate((tmpH,tmpV),axis = 0)
            
            # gen.plot_constellations(rx['{}_cplx_stuffed'.format(what)],polar = 'both')

    return rx

# %%
def save_estimations(rx):

    rx["H_est_l"].append(rx["h_est"].tolist())

    return rx

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


