# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          :
#   Date            : 2023-03-04
#   Version         : 1.0.0
#   Licence         : cc-by-nc-sa
#                     Attribution - Non-Commercial - Share Alike 4.0 International
# 
# ----- Main idea -----
# ----- INPUTS -----
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
#   Author              : 
#   Author contact      : 
#   Affiliation         :
#   Date                : 
#   Title of program    : 
#   Code version        : 
#   Type                :
#   Web Address         :
# ---------------------------------------------
# %%

import numpy as np
import matplotlib.pyplot as plt
import lib_misc as misc



#%%
# =============================================================================
# rx_load_ase
# =============================================================================
def load_ase(tx,rx):
    
    # noise added at the receiver:
    # When we set a value of SNR, it will be divided into 4 equal noise components: HI,HQ,VI,VQ
    # SNRdB = 20, P_rx_sig = 1 [mW]
    # ==> P_noise_tot = 0.01   [mW]
    # ==> P_noise_AB  = 0.0025 [mW]   ---   A in {H,V}, B in {I,Q}    

    SNR             = 10**(rx["SNRdB"]/10)
    rx["P_rx_sig"]  = np.mean(np.abs(rx["sig_cplx"])**2)    # total power in X+Y polarisations [W]
    rx["P_noise"]   = rx["P_rx_sig"]/2/SNR
    sigma_n         = np.sqrt(rx["P_noise"]*tx["Nsps"])
    
    randn_I         = np.random.randn(tx["Npolars"],tx["Nsamp_rx_tmp"])
    randn_Q         = np.random.randn(tx["Npolars"],tx["Nsamp_rx_tmp"])
    randn_IQ        = randn_I+1j*randn_Q
    Noise           = sigma_n*randn_IQ    
    rx["sig_cplx"]  = rx["sig_cplx"] + Noise
    
    # conversion into desired sizes : we remove the excess symbols
    RHI             = np.real(rx["sig_cplx"][0][:tx["Nsamp_rx_tmp"]])
    RHQ             = np.imag(rx["sig_cplx"][0][:tx["Nsamp_rx_tmp"]])
    RVI             = np.real(rx["sig_cplx"][1][:tx["Nsamp_rx_tmp"]])
    RVQ             = np.imag(rx["sig_cplx"][1][:tx["Nsamp_rx_tmp"]])
    
    rx["sig_real"]  = misc.my_tensor(np.array([[RHI,RHQ],[RVI,RVQ]]))
    

    return rx