# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.2.3
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
#   1.1.1 (2024-03-27) - gen_phase_noise: new slicing mode
#                      - [NEW] set_phis
#   1.2.0 (2024-05-29) - load_ase, load_phase_noise --- replaces 1.1.1 slicing mode from frame-wise 
#                           to batch-wise
#                      - [REMOVED] set_phis, included into gen_phase_noise
#   1.2.1 (2024-07-01) - load_ase: SNR -> tx['SNR']
#   1.2.2 (2024-07-02) - gen_phase_noise: tx["PhiLaw"]["law"]  == "linewidth",
#                           corrected the variance
#   1.2.3 (2024-07-10) - naming normalisation (*frame*-> *Frame*).
#                        along with main (1.4.3)
#
# ----- MAIN IDEA -----
#   Generation and management of phase noise and ASE noise in optical telecommunication systems
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books
#   Authors             : 
#   Title               :
#   Journal/Editor      : 
#   Volume - N°         : 
#   Date                :
#   DOI/ISBN            :
#   Pages               :
#  ----------------------
#   Functions
#   Author              :
#   Contact             :
#   Affiliation         :
#   Date                :
#   Title of program    :
#   Code version        :
#   Type                :
#   Web Address         :
# ---------------------------------------------
# %%

#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import lib_misc as misc
from lib_misc import KEYS as keys
import lib_plot as plot
import torch

from lib_matlab import clc
pi = np.pi

#%% ===========================================================================
# --- CONTENTS
# =============================================================================
# - gen_phase_noise             --- called in : processing.init_processing
# - load_ase                    --- called in : txdsp.transmitter
# - load_phase_noise            --- called in : txdsp.transmitter
# - set_phis                    --- called in : processing.init_processing
# =============================================================================


#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================

#%%
def gen_phase_noise(tx,rx):

    if tx['flag_phase_noise'] == 1:
            
        if 'PhiLaw' not in tx:
            tx["PhiLaw"] = dict()
    
        if "kind" not in tx["PhiLaw"]:
            tx["PhiLaw"]["kind"]    = "Rwalk"
            tx["PhiLaw"]["law"]     = "linewidth"
    
        if 'PhaseNoise_mode' not in tx:
            tx['PhaseNoise_mode'] = 'batch-wise'
            

        # --------------------------------------- BATCH-WISE
        if tx['PhaseNoise_mode'] == "batch-wise":

            tmp_phase_noise = np.zeros((1,rx['NFrames'],rx['NBatchFrame']))
            if tx["PhiLaw"]['kind']         == 'Rwalk':

                tmp_last_phase  = 0
                
                # set angles by batches
                for k in range(rx['NFramesChannel']):

                    if tx["PhiLaw"]["law"]      == "linewidth":
                        tx['VAR_Phase']         = 2*pi*tx['dnu']/tx['fs']
                        
                        noise_tmp               = np.random.normal(0,np.sqrt(tx['VAR_Phase']),rx["NBatchFrame"])
                        tmp_phase_noise[0,k,:]  = tmp_last_phase + np.cumsum(noise_tmp)
                        tmp_last_phase          = tmp_phase_noise[0,k,-1]


                # circshift to put the zeros phases where they should be
                tmp_phase_noise[0]      = np.roll(tmp_phase_noise[0], -rx['NFramesChannel'], axis = 0) # axis 0 = rows
                

            else:
                if tx["PhiLaw"]["law"]      == "lin":
                    PhiStart                = tx["PhiLaw"]["Start"]            # [rad]
                    PhiEnd                  = tx["PhiLaw"]["End"]
                    tx["PhiLaw"]["Sph"]     = (PhiEnd-PhiStart)/tx["NsampFrame"]

                    tmp_training            = np.linspace(PhiStart,PhiEnd,rx["NBatchFrame"]*rx['NFramesChannel'])
                    tmp_channel             = np.zeros((rx["NBatchFrame"]*rx['NFramesTraining']))

                    tmp                     = np.concatenate((tmp_channel,tmp_training),axis = 0)
                    tmp_phase_noise[0]      = np.reshape(tmp,(rx['NFrames'],-1))
            

            tx["PhaseNoise_unique"] = tmp_phase_noise[0].squeeze()
            tmp_pn                  = np.repeat(tmp_phase_noise[0],rx['NsampBatch'],axis=1)
            tmp_pn_rs               = np.reshape(tmp_pn,(rx['NFrames'],-1))
            
            Nexcess_symbs           = abs(tx['NsampFrame']-rx['NsampBatch']*rx['NBatchFrame'])
            for k in range(rx['NFrames']):
                tmp                 = tmp_pn_rs[k,-Nexcess_symbs-1:-1]
                tx["PhaseNoise"][0,:,k] = np.concatenate((tmp_pn_rs[k], tmp),axis = 0)
    
            tx["PhaseNoise"][1]     = tx["PhaseNoise"][0]
            
            # phase noise estimation after filtering
            tx['pn_fil_losses']     = np.zeros((rx['NFrames'],tx['pn_filt_par']['niter_max']))
            
            # for checking
            # for frame in range(rx['FrameChannel']-1,rx['FrameChannel']+2):
            #     plt.plot(tx['PhaseNoise'][0,:,frame], label = 'frame No {}'.format(frame))
            
            # plt.legend()


        # --------------------------------------- SAMP-WISE
        else:
            tx["PhaseNoise"]                = np.zeros((2,tx["NsampTot"]))
            
            if tx["PhiLaw"]['kind']         == 'Rwalk':
                if tx["PhiLaw"]["law"]      == "linewidth":
                    tx['VAR_Phase']         = 2*pi*tx['dnu']/tx['fs']
    
                    noise_tmp               = np.random.normal(0,np.sqrt(tx['VAR_Phase']),tx["Nsamp_PhaseNoise"])
    
                    tx['PhaseNoise'][0,tx['Nsamp_training']:]   = np.cumsum(noise_tmp)
                    tx['PhaseNoise'][1,tx['Nsamp_training']:]   = np.cumsum(noise_tmp)
                    
            elif tx["PhiLaw"]['kind']       != "Rwalk":
                if tx["PhiLaw"]["law"]      == "lin":
                    PhiStart                = tx["PhiLaw"]["Start"]
                    PhiEnd                  = tx["PhiLaw"]["End"]
                    tx["PhiLaw"]["Sph"]     = (PhiEnd-PhiStart)/tx["Nsamp_PhaseNoise"]
                    
                    tx["PhaseNoise"][0,tx['Nsamp_training']:]   = np.linspace(PhiStart,PhiEnd,tx["Nsamp_PhaseNoise"])
                    tx["PhaseNoise"][1,tx['Nsamp_training']:]   = np.linspace(PhiStart,PhiEnd,tx["Nsamp_PhaseNoise"])
    
    
            tmp                 = np.zeros((2,tx['NsampFrame'],rx['NFrames'])).astype(np.float32)
            for k in range(rx['NFrames']):
                tmp[:,:,k]      = tx['PhaseNoise'][:,k*tx["NsampFrame"]:(k+1)*tx["NsampFrame"]]
    
            tx['PhaseNoise']    = tmp




    tx                      = misc.sort_dict_by_keys(tx)
    return tx



#%%
def load_ase(tx,rx):
    
    if rx['Frame'] == rx['FrameChannel']:
        print('loading ASE @ TX\n')
    # noise added at the transmitter:
    # When we set a value of SNR, it will be divided into 4 equal noise components: HI,HQ,VI,VQ
    # SNRdB = 20, P_rx_sig = 1 [mW]
    # ==> P_noise_tot = 0.01   [mW]
    # ==> P_noise_AB  = 0.0025 [mW]   ---   A in {H,V}, B in {I,Q}

    tx['SNR']           = 10**(tx['SNRdB']/10)
    
    tx_sig_cplx         = np.zeros((tx['Npolars'],tx['NsampFrame']),dtype = np.complex64)
    tx_sig_cplx[0]      = np.array(tx['sig_real'][0]+1j*tx['sig_real'][1])
    tx_sig_cplx[1]      = np.array(tx['sig_real'][2]+1j*tx['sig_real'][3])
    tx["P_tx_sig"]      = np.mean(np.abs(tx_sig_cplx)**2)    # total power in X+Y polarisations [W]

    tx["P_tx_noise"]    = tx["P_tx_sig"]/2/tx['SNR']
    sigma_n             = np.sqrt(tx["P_tx_noise"]*tx["Nsps"])

    randn_I             = np.random.randn(tx["Npolars"],tx["NsampFrame"]).astype(np.float32)
    randn_Q             = np.random.randn(tx["Npolars"],tx["NsampFrame"]).astype(np.float32)

    randn_IQ            = (randn_I+1j*randn_Q).astype(np.complex64)
    Noise               = sigma_n*randn_IQ
    tx_sig_cplx         = tx_sig_cplx + Noise

    tx['sig_real'][0]   = torch.tensor(np.real(tx_sig_cplx[0]))
    tx['sig_real'][1]   = torch.tensor(np.imag(tx_sig_cplx[0]))
    tx['sig_real'][2]   = torch.tensor(np.real(tx_sig_cplx[1]))
    tx['sig_real'][3]   = torch.tensor(np.imag(tx_sig_cplx[1]))
    
    # plot.constellation(tx["sig_cplx"][0],title = "TX ase loading)

    return tx



#%%
def load_phase_noise(tx,rx,*varargin):

    if rx['Frame'] == rx['FrameChannel']:
        print('loading PhaseNoise @ TX\n')
        
    exp_phase           = np.exp(1j*tx['PhaseNoise'][:,:,rx['Frame']])
    
    tx_sig_cplx         = np.zeros((tx['Npolars'],tx['NsampFrame']),dtype = np.complex64)
    tx_sig_cplx[0]      = np.array(tx['sig_real'][0]+1j*tx['sig_real'][1])
    tx_sig_cplx[1]      = np.array(tx['sig_real'][2]+1j*tx['sig_real'][3])
    tx_sig_cplx         = np.multiply(tx_sig_cplx,exp_phase)

    tx['sig_real'][0]   = torch.tensor(np.real(tx_sig_cplx[0]))
    tx['sig_real'][1]   = torch.tensor(np.imag(tx_sig_cplx[0]))
    tx['sig_real'][2]   = torch.tensor(np.real(tx_sig_cplx[1]))
    tx['sig_real'][3]   = torch.tensor(np.imag(tx_sig_cplx[1]))

    if len(varargin) != 0 and 'pn time trace' in varargin:
        for k in range(rx['NFrames']):
            pns = round(tx['PhaseNoise'][0,0,k]*180/pi,3)
            pne = round(tx['PhaseNoise'][0,-1,k]*180/pi,3)
            
            plt.figure()
            plt.plot(tx['PhaseNoise'][0,:,k]*180/pi)
            plt.title(f"frame {k}, pn-s = {pns}, pn-e = {pne}")
            plt.show()
            
    if len(varargin) != 0 and 'pn const' in varargin:
        plot.constellations(tx["sig_cplx"],title = 'phase noise laoding')

    tx              = misc.sort_dict_by_keys(tx)

    return tx



