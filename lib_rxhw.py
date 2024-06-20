# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.0.2
#   Date            : 2023-04-01
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2023-04-04) - creation
#   1.0.2 (2023-04-01) - cleaning
#
# ----- MAIN IDEA -----
#   Basic functions and setup for the initial phase of the project
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
import torch

#%% ===========================================================================
# --- CONTENTS
# =============================================================================
# - load_ase
# =============================================================================


#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================
#%%
def load_ase(tx,rx):
    
    # noise added at the receiver:
    # When we set a value of SNR, it will be divided into 4 equal noise components: HI,HQ,VI,VQ
    # SNRdB = 20, P_rx_sig = 1 [mW]
    # ==> P_noise_tot = 0.01   [mW]
    # ==> P_noise_AB  = 0.0025 [mW]   ---   A in {H,V}, B in {I,Q}    

    rx['SNR']               = 10**(rx['SNRdB']/10)
    rx["noise_var"]         = torch.full((2,), tx['pow_mean']/rx['SNR']/2)

    # SNR                 = 10**(rx['SNRdB']/10)

    rx_sig_cplx         = np.zeros((tx['Npolars'],tx['NsampFrame']),dtype = np.complex64)
    rx_sig_cplx[0]      = np.array(rx['sig_real'][0]+1j*rx['sig_real'][1])
    rx_sig_cplx[1]      = np.array(rx['sig_real'][2]+1j*rx['sig_real'][3])

    rx["P_rx_sig"]      = np.mean(np.abs(rx_sig_cplx)**2)    # total power in X+Y polarisations [W]
    rx["P_noise"]       = rx["P_rx_sig"]/2/rx['SNR']

    sigma_n             = np.sqrt(rx["P_noise"]*tx["Nsps"])

    randn_I             = np.random.randn(tx["Npolars"],tx["NsampFrame"]).astype(np.float32)
    randn_Q             = np.random.randn(tx["Npolars"],tx["NsampFrame"]).astype(np.float32)

    randn_IQ            = (randn_I+1j*randn_Q).astype(np.complex64)
    rx['Noise']         = sigma_n*randn_IQ    
    rx_sig_cplx         = rx_sig_cplx + rx['Noise']

    rx['sig_real'][0]   = torch.tensor(np.real(rx_sig_cplx[0]))   # HI
    rx['sig_real'][1]   = torch.tensor(np.imag(rx_sig_cplx[0]))   # HQ
    rx['sig_real'][2]   = torch.tensor(np.real(rx_sig_cplx[1]))   # VI
    rx['sig_real'][3]   = torch.tensor(np.imag(rx_sig_cplx[1]))   # VQ

    rx                  = misc.sort_dict_by_keys(rx)


    return rx