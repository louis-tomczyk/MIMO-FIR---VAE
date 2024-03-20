# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2023-03-04 (1.0.0) - creation
#                     2023-03-16 (1.0.1) - cleaning
#   Date            : 2023-03-16 (1.0.2) - cleaning
#   Version         : 1.0.1
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

import numpy as np
import matplotlib.pyplot as plt
import torch

import lib_misc as misc
import lib_txhw as txhw
import lib_general as gen

pi = np.pi

#%%
def get_constellation(tx,rx):
    
    # définition des paramètres pour chaque constellation
    constellations = {}
    params = {
        '4QAM':    (2, [-1, 1]),
        '16QAM':   (4, [-3, -1, 1, 3]),
        '64QAM':   (8, [-7, -5, -3, -1, 1, 3, 5, 7]),
        '256QAM':  (16, [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
    }
    
    # création des constellations disponibles
    for modfor, (side, amplitudes) in params.items():
        points = []
        for i in range(side):
            for j in range(side):
                points.append(amplitudes[i] + 1j * amplitudes[j])
        constellations[modfor] = np.array(points)
    
    # création de ma constellation utilisée
    # norm factor:
    # 4qam: 1 --- 16 qam: sqrt(10) --- 64qam: 6.48 --- 256qam: 13.03
    constellation   = constellations[tx["mod"]]
    norm_factor     = np.sqrt(np.mean(np.abs(constellation)**2))
    constellation   = constellation/norm_factor

    # misc.plot_complex_constellation(constellation)
    
    # -----------------------------------------------------------------------------------------    
    # IN    = NU (float) 
    # OUT   = P (np.array) probability matrix of the sent symbols
    # -----------------------------------------------------------------------------------------

    # extract real amplitudes

    N_amps          = int(np.sqrt(len(constellation.real)))         # number of ASK levels
    amps            = misc.my_tensor(constellation.real[::N_amps])   # amplitude levels
    sc              = torch.min(np.abs(amps))                       # scaling factor for having lowest level equal 1
    nu              = tx["nu"]
    nu_sc           = tx["nu"]/sc**2                                # re-scaled shaping factor

    # probabilities of the amlitude levels
    P               = np.exp(-nu*np.abs(amps/sc)**2)
    P               = P / torch.sum(P)  

    P               = np.expand_dims(P, axis=0)
    T               = np.dot(P.T,P)
    P               = misc.my_tensor(P.reshape(-1))


    # misc.imagesc(np.log10(T)) # plot the probability matrix
    # -----------------------------------------------------------------------------------------    
    # OTHERS
    # -----------------------------------------------------------------------------------------
    
    # (optional) entropy of the modulation format
    # max H_p = 2*ln (N_amp), 'ln' = natural log (logarithme népérien) 
    # 64 qam => H_P = 4.1588...
    # H_P = -np.sum(np.log(T)*T)
    
    # mean power of the constellation
    pow_mean = np.sum(T.reshape(-1)* np.abs(constellation)**2)
    
    tx["amps"]              = amps                      # amplitude levels of the modulation format
    tx["const_norm_factor"] = norm_factor   
    tx["constellation"]     = constellation
    tx["N_amps"]            = N_amps                    # number of positive amplitud
    tx["nu_sc"]             = nu_sc                     # scaling factor for the PCS
    tx["pow_mean"]          = pow_mean                  # mean power of the constellation
    tx["prob_amps"]         = P                         # probabilities of the amplitude levels
    
    return tx,rx
    

#%%
# ============================================================================================ #
# Generate the data
# ============================================================================================ #
def data_generation(tx,rx):
    
    # If P == uniform, the plt.plot(data[0]) will be looking
    # like filled rectangle for instance:
    #
    #     p = (np.ones((1,len(amps)))*1/len(amps)).reshape(-1)
    #
    # DATA = 
    #   [
    #       data array with pol-H channel-I, shape = (1,Nconv)
    #       data array with pol-H channel-Q, shape = (1,Nconv)
    #       data array with pol-V channel-I, shape = (1,Nconv)
    #       data array with pol-V channel-Q, shape = (1,Nconv)
    #   ] =
    #   [
    #       sHI[0],sHI[1],...,sHI[Nconv-1]
    #       sHQ[0],sHQ[1],...,sHQ[Nconv-1]
    #       sVI[0],sVI[1],...,sVI[Nconv-1]
    #       sVQ[0],sVQ[1],...,sVQ[Nconv-1]
    #   ]
    # DATA_I = 
    #   [
    #       sHI[0],sHI[1],...,sHI[Nconv-1]
    #       sVI[0],sVI[1],...,sVI[Nconv-1]
    #   ]

    #1 data arrays parameters
    # # Nconv = NSymbFrame + some extra symbols necessary for
    # # edges management
    # tx["Nconv"]     = rx["NSymbFrame"]+tx["Ntaps"]+1
    # tx["Nsamp_up"]  = tx["Nsps"]*(tx["Nconv"]-1)+1
    
    #2 data generation
    # draw randomly amplitude values from AMPS using the law 
    # of probability used in the Probabilistic Constellation
    # Shaping.
    
    data            = np.random.default_rng().choice(tx["amps"],
                                                     (2*tx["Npolars"],tx["Nconv"]),
                                                     p=np.round(tx["prob_amps"],5))
    data_I          = data[0::tx["Npolars"],:] # skip using '::<number of rows to skip>
    data_Q          = data[1::tx["Npolars"],:]

    # sps-upsampled signal by zero-insertion
    # louis, do not remove the type definition
    tx["sig_cplx_up"]                   = np.zeros((tx["Npolars"],tx["Nsamp_up"]),dtype=np.complex128)
    tx["sig_cplx_up"][:,::tx["Nsps"]]   = data_I + 1j*data_Q

    mask_start      = int(tx["Ntaps"]/2)
    mask_end        = rx["NSymbFrame"]+tx["NSymbTaps"]-1
    mask_indexes    = list(range(mask_start,mask_end))
    
    THI             = data[0][mask_indexes]
    THQ             = data[1][mask_indexes]
    TVI             = data[2][mask_indexes]
    TVQ             = data[3][mask_indexes]
    
    # louis, don't try to convert into complex before concatenating, keep it after
    TH              = THI+1j*THQ
    TV              = TVI+1j*TVQ

    tx["symb_real"] = misc.my_tensor(np.array([[THI,THQ],[TVI,TVQ]]))
    tx["symb_cplx"] = misc.my_tensor(np.array([TH,TV]),dtype=torch.complex128)
    
    tx              = misc.sort_dict_by_keys(tx)
    return tx


# ============================================================================================ #
# root raised-cosine filter
# ============================================================================================ #
def rrcfir(tx):
    
    # symbol-normalised time array that contains len(t) = T*sps+1 elements
    # Nsps = fs*Tsymb => t échantilloné à fs et normalisé par le temps symbole
    t       = np.arange(-tx["NSymbTaps"]/2 +1/tx["Nsps"], tx["NSymbTaps"]/2, 1/tx["Nsps"])
    
    # filter = num/den
    den     = pi*t*(1-(4*tx["RollOff"]*t)**2)
    num     = np.sin(pi*t*(1-tx["RollOff"])) + 4*tx["RollOff"]*t*np.cos(pi*t*(1+tx["RollOff"]))
    
    # avoiding division by zero
    # 1/ find   where den == 0
    # 2/ remove where den == 0
    # 3/ make the filter

    i0s     = np.where(den==0)[0]
    denNot0 = misc.remove_using_index(i0s, den)
    numNot0 = misc.remove_using_index(i0s, num)
    hNot0   = numNot0/denNot0
    h       = hNot0
    
    # particular values of the RRC
    i0      = np.where(t==0)[0]
    ipm     = np.where(np.abs(t)==1/4/tx["RollOff"])[0]
    h0      = 1 + tx["RollOff"]*(4/pi - 1)
    hpm     = tx["RollOff"]/np.sqrt(2)*((1+2/pi)*np.sin(pi/4/tx["RollOff"])
                                     +(1-2/pi)*np.cos(pi/4/tx["RollOff"]))

    h       = np.insert(h,i0-1,h0)
    h       = np.insert(h,ipm[0],hpm)
    h       = np.insert(h,ipm[1],hpm)
    
    # normalisation using Frobenius norm
    tx["h_pulse"] = h/np.linalg.norm(h)
    
    tx      = misc.sort_dict_by_keys(tx)
    
    return tx


# ============================================================================================ #
# does the shaping with the RRC filter
# ============================================================================================ #
def data_shaping(tx):

    # # The "valid" mode for convolution already shrinks the length
    # tx["Nsamp_gross"]  = tx["Nsamp_up"]-(tx["Ntaps"]-1) # = 10,377-13+1=10,365

    # 0 == pol H ------- 1 == pol V
    tx["sig_cplx"]      = np.zeros((tx["Npolars"],tx["Nsamp_gross"] ),dtype=np.complex128)
    tx["sig_cplx"][0,:] = np.convolve(tx["sig_cplx_up"][0,:], tx["h_pulse"], mode = 'valid')
    tx["sig_cplx"][1,:] = np.convolve(tx["sig_cplx_up"][1,:], tx["h_pulse"], mode = 'valid')

    tx                  = misc.sort_dict_by_keys(tx)
    
    return tx


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def transmitter(tx,rx):

    if tx["mode"] == "self":
        tx          = data_generation(tx,rx)
        tx          = rrcfir(tx)
        tx          = data_shaping(tx)
        
            
        if rx["Frame"] == rx["FrameRndRot"]:
            print('loading ASE and Phase Noise at TX\n')
            tx      = txhw.load_ase(tx)
            tx      = txhw.load_phase_noise(tx,rx)
        
    elif tx["mode"] == "ext":
        tx["Nsamp"] = tx["sig_real"].shape[2]
    
    tx              = misc.sort_dict_by_keys(tx)
    
    # if rx['Frame'] == 0:
    #     import time
    #     gen.plot_const_2pol(tx['sig_cplx'],"tx")
    #     time.sleep(5)
    #     plt.show()
    return tx

