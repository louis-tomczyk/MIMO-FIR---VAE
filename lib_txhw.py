# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2023-03-04 (1.0.0) - creation
#   Date            : 2023-03-20 (1.1.0) - gen_phase_noise
#   Version         : 1.1.0
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
    
    randn_I         = np.random.randn(tx["Npolars"],tx["Nsamp_gross"])
    randn_Q         = np.random.randn(tx["Npolars"],tx["Nsamp_gross"])
    
    
    randn_IQ        = randn_I+1j*randn_Q
    Noise           = sigma_n*randn_IQ    
    tx["sig_cplx"]  = tx["sig_cplx"] + Noise
    
    # misc.plot_const_1pol(tx["sig_cplx"][0],'ase laoding')
    
    # conversion into desired sizes : we remove the excess symbols
    # THI             = np.real(tx["sig_cplx"][0][:tx["Nsamp_net"]])
    # THQ             = np.imag(tx["sig_cplx"][0][:tx["Nsamp_net"]])
    # TVI             = np.real(tx["sig_cplx"][1][:tx["Nsamp_net"]])
    # TVQ             = np.imag(tx["sig_cplx"][1][:tx["Nsamp_net"]])
    
    THI             = np.real(tx["sig_cplx"][0][:tx["Nsamp_gross"]])
    THQ             = np.imag(tx["sig_cplx"][0][:tx["Nsamp_gross"]])
    TVI             = np.real(tx["sig_cplx"][1][:tx["Nsamp_gross"]])
    TVQ             = np.imag(tx["sig_cplx"][1][:tx["Nsamp_gross"]])
    
    tx["sig_real"]  = misc.my_tensor(np.array([[THI,THQ],[TVI,TVQ]]))
    
    return tx

#%%
# =============================================================================
# =============================================================================

def load_phase_noise(tx,rx):

        
    tx["DeltaPhi"]  = np.diff(tx["PhaseNoise"])    
    tx["sig_cplx"]  = np.multiply(tx["sig_cplx"],np.exp(1j*2*np.pi*tx['PhaseNoise'][:,:,rx['Frame']]))

    
    # plt.figure()
    # plt.plot(tx['PhaseNoise'][0])
    # plt.plot(tx['PhaseNoise'][1])
    # misc.plot_const_2pol(tx["sig_cplx"],'phase noise laoding')
    

    
    THI             = np.real(tx["sig_cplx"][0][:tx["Nsamp_gross"]])
    THQ             = np.imag(tx["sig_cplx"][0][:tx["Nsamp_gross"]])
    TVI             = np.real(tx["sig_cplx"][1][:tx["Nsamp_gross"]])
    TVQ             = np.imag(tx["sig_cplx"][1][:tx["Nsamp_gross"]])
    
    tx["sig_real"]  = misc.my_tensor(np.array([[THI,THQ],[TVI,TVQ]]))
    tx              = misc.sort_dict_by_keys(tx)
    
    return tx



#%%
# =============================================================================
# =============================================================================
def gen_phase_noise(tx,rx):
    
    if 'PhiLaw' not in tx:
        tx["PhiLaw"] = dict()

    if "kind" not in tx["PhiLaw"]:
        tx["PhiLaw"]["kind"]    = "Rwalk"
        tx["PhiLaw"]["law"]     = "linewidth"
        
    tx["PhaseNoise"]            = np.zeros((2,tx["Nsamp_gross"], rx['Nframes']))
    
    for k in range(rx['FrameRndRot'],rx['Nframes']):
        
        if tx["PhiLaw"]['kind']         == 'Rwalk':
            if tx["PhiLaw"]["law"]      == "linewidth":
                tx['VAR_Phase']         = np.sqrt(2*np.pi*tx['dnu']/tx['fs'])
                
                noise_tmp               = np.random.normal(0,tx['VAR_Phase'],tx["Nsamp_gross"])    
                tx['PhaseNoise'][0,:,k] = np.cumsum(noise_tmp)
                tx['PhaseNoise'][1,:,k] = np.cumsum(noise_tmp)
        else:
            if tx["PhiLaw"]["law"]      == "lin":
                PhiStart                = tx["PhiLaw"]["Start"]
                PhiEnd                  = tx["PhiLaw"]["End"]
                tx["PhiLaw"]["slope"]   = (PhiEnd-PhiStart)/tx["Nsamp_gross"]
                tx["PhaseNoise"][0,:,k] = np.linspace(PhiStart,PhiEnd,tx["Nsamp_gross"])
                tx["PhaseNoise"][1,:,k] = np.linspace(PhiStart,PhiEnd,tx["Nsamp_gross"])
    
    
    
    tx                          = misc.sort_dict_by_keys(tx)
    return tx
