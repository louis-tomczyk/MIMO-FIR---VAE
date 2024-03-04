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


# ============================================================================================ #
# tx_load_ase
# ============================================================================================ #
def load_ase(tx):
    
    # noise added at the transmitter:
    # When we set a value of SNR, it will be divided into 4 equal noise components: HI,HQ,VI,VQ
    # SNRdB = 20, P_rx_sig = 1 [mW]
    # ==> P_noise_tot = 0.01   [mW]
    # ==> P_noise_AB  = 0.0025 [mW]   ---   A in {H,V}, B in {I,Q}    


    SNR             = 10**(tx["SNRdB"]/10)
    tx["P_tx_sig"]  = np.mean(np.abs(tx["sig_cplx"])**2)    # total power in X+Y polarisations [W]
    tx["P_tx_noise"]= tx["P_tx_sig"]/2/SNR
    sigma_n         = np.sqrt(tx["P_tx_noise"]*tx["Nsps"])
    
    randn_I         = np.random.randn(tx["Npolars"],tx["Nsamp_rx_tmp"])
    randn_Q         = np.random.randn(tx["Npolars"],tx["Nsamp_rx_tmp"])
    randn_IQ        = randn_I+1j*randn_Q
    Noise           = sigma_n*randn_IQ    
    tx["sig_cplx"]  = tx["sig_cplx"] + Noise
    
    # misc.plot_const_1pol(tx["sig_cplx"][0],'ase laoding')
    
    # conversion into desired sizes : we remove the excess symbols
    THI             = np.real(tx["sig_cplx"][0][:tx["Nsamp_rx_tmp"]])
    THQ             = np.imag(tx["sig_cplx"][0][:tx["Nsamp_rx_tmp"]])
    TVI             = np.real(tx["sig_cplx"][1][:tx["Nsamp_rx_tmp"]])
    TVQ             = np.imag(tx["sig_cplx"][1][:tx["Nsamp_rx_tmp"]])
    
    tx["sig_real"]  = misc.my_tensor(np.array([[THI,THQ],[TVI,TVQ]]))
    
    return tx

# =============================================================================
# rx_load_ase
# =============================================================================

def load_phase_noise(tx):

    tx['VAR_Phase']     = np.sqrt(2*np.pi*tx['linewidth']/tx['fs'])
    tx['PhaseNoise']    = np.zeros((2,tx['Nsamp_rx_tmp'])) 
    
    noise_tmp               = np.random.normal(0,tx['VAR_Phase'],tx['Nsamp_rx_tmp'])
    tx['PhaseNoise'][0,:]   = np.cumsum(noise_tmp)
    tx['PhaseNoise'][1,:]   = np.cumsum(noise_tmp)
    tx["sig_cplx"]          = np.multiply(tx["sig_cplx"],np.exp(1j*2*np.pi*tx['PhaseNoise']))

    
    # plt.figure()
    # plt.plot(tx['PhaseNoise'][0])
    # plt.plot(tx['PhaseNoise'][1])
    # misc.plot_const_2pol(tx["sig_cplx"],'phase noise laoding')
    
    # conversion into desired sizes : we remove the excess symbols
    THI             = np.real(tx["sig_cplx"][0][:tx["Nsamp_rx_tmp"]])
    THQ             = np.imag(tx["sig_cplx"][0][:tx["Nsamp_rx_tmp"]])
    TVI             = np.real(tx["sig_cplx"][1][:tx["Nsamp_rx_tmp"]])
    TVQ             = np.imag(tx["sig_cplx"][1][:tx["Nsamp_rx_tmp"]])
    
    tx["sig_real"]  = misc.my_tensor(np.array([[THI,THQ],[TVI,TVQ]]))
    
    return tx