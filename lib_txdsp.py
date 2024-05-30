# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2024-03-04 (1.0.0) - creation
#                   : 2024-04-03 (1.0.3) - cleaning
#   Date            : 2024-05-27 (1.0.4) - get_constellation: inspiration from [C2]
#   Version         : 1.0.4
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute - place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source - include copyright - state changes - include license
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
#   Author              : [C1] Vincent LAUINGER
#   Author contact      : vincent.lauinger@kit.edu
#   Affiliation         : Communications Engineering Lab (CEL)
#                           Karlsruhe Institute of Technology (KIT)
#   Date                : 2022-06-15
#   Title of program    : 
#   Code version        : 
#   Type                : source code
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
#%%
import numpy as np
import torch
import sys
import lib_misc as misc
import lib_general as gen
import lib_txhw as txhw
import lib_matlab as mb
pi = np.pi



#%% ===========================================================================
# --- CONTENTS ---
# =============================================================================
# - data_generation         --- called in : transmitter
# - data_shaping            --- called in : transmitter
# - get_constellation       --- called in : processing.init_processing
# - set_Nsymbols            --- called in : main.process_data
# - shaping_filter          --- called in : transmitter
# - transmitter             --- called in : processing.processing
# =============================================================================



#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================

#%%
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
    tx["sig_cplx_up"]                   = np.zeros((tx["Npolars"],tx["Nsamp_up"]),dtype=np.complex64)
    tx["sig_cplx_up"][:,::tx["Nsps"]]   = data_I + 1j*data_Q

    mask_start      = int(tx["Ntaps"]/2)
    mask_end        = rx["NSymbFrame"]+tx["NSymbTaps"]-1
    mask_indexes    = list(range(mask_start,mask_end))

    THI             = data[0][mask_indexes]
    THQ             = data[1][mask_indexes]
    TVI             = data[2][mask_indexes]
    TVQ             = data[3][mask_indexes]

    tx["Symb_real"] = misc.my_tensor(np.array([[THI,THQ],[TVI,TVQ]]))

    tx              = misc.sort_dict_by_keys(tx)
    return tx

#%%
def data_shaping(tx):

    # # The "valid" mode for convolution already shrinks the length
    # tx["NsampFrame"]  = tx["Nsamp_up"]-(tx["Ntaps"]-1) # = 10,377-13+1=10,365
    # 0 == pol H ------- 1 == pol V

    h_pulse             = tx['hmatrix'][0]
    tx["sig_real"]      = np.zeros((tx["Npolars"]*2,tx["NsampFrame"]),dtype=np.float32)

    tmp                 = np.convolve(tx["sig_cplx_up"][0,:], h_pulse, mode = 'valid')    
    tx["sig_real"][0]   = np.real(tmp)
    tx["sig_real"][1]   = np.imag(tmp)

    tmp                 = np.convolve(tx["sig_cplx_up"][1,:], h_pulse, mode = 'valid')
    tx["sig_real"][2]   = np.real(tmp)
    tx["sig_real"][3]   = np.imag(tmp)

    del tx["sig_cplx_up"], tmp


    tx                  = misc.sort_dict_by_keys(tx)

    return tx


#%%
def get_constellation(tx,rx):
    
    # définition des paramètres pour chaque constellation
    constellations = {}
    params = {
        '4QAM'  :   (2, [-1, 1]),
        '16QAM' :   (4, [-3, -1, 1, 3]),
        '64QAM' :   (8, [-7, -5, -3, -1, 1, 3, 5, 7]),
        '256QAM':   (16, [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
    }
    
    Nradii = {
        '4QAM'  :   (1, [4]),
        '16QAM' :   (3, [4, 8, 4]),
        '64QAM' :   (9, [4, 8, 4, 8, 8, 12, 8, 8, 4])
    }
    
    # création des constellations disponibles
    for modfor, (side, amplitudes) in params.items():
        points = []
        for i in range(side):
            for j in range(side):
                points.append(amplitudes[i] + 1j * amplitudes[j])
        constellations[modfor] = np.array(points)


    # norm factor: sqrt(2*(M-1)/3)
    # 4qam: sqrt(2) --- 16 qam: sqrt(10) --- 64qam: sqrt(42) --- 256qam: sqrt(170)
    tx["const_affixes"]   = constellations[tx["mod"]]

    rings           = np.unique(abs(tx["const_affixes"]))
    Prob_symb_tmp   = np.exp(-tx['nu']*rings**2)
    Prob_ring_tmp   = Prob_symb_tmp*Nradii[tx['mod']][1]
    Prob_symb       = Prob_symb_tmp/sum(Prob_ring_tmp)
    Prob_ring       = Prob_symb*Nradii[tx['mod']][1]
    
    Symb_Probs_tmp  = np.exp(-tx['nu']*abs(tx["const_affixes"]))
    Symb_Probs      = Symb_Probs_tmp/sum(Symb_Probs_tmp)
    
    norm_factor     = np.sqrt(np.mean(np.abs(tx["const_affixes"])**2))
    constellation   = tx["const_affixes"]/norm_factor

    # misc.plot_complex_constellation(constellation)
    
    # -----------------------------------------------------------------------------------------    
    # IN    = NU (float) 
    # OUT   = P (np.array) probability matrix of the sent symbols
    # -----------------------------------------------------------------------------------------

    # extract real amplitudes

    N_amps          = int(np.sqrt(len(constellation.real)))         # number of ASK levels
    amps            = constellation.real[::N_amps] # amplitude levels
    sc              = min(abs(amps))                       # scaling factor for having lowest level equal 1
    nu_sc           = tx['nu']/(sc**2)

    # probabilities of the amlitude levels
    P               = np.exp(-tx["nu"]*np.abs(amps/sc)**2)
    P               = P / sum(P)  

    P               = np.expand_dims(P, axis=0)
    T               = np.dot(P.T,P)
    P               = P.reshape(-1)


    # mb.imagesc(np.log10(T)) # plot the probability matrix
    # -----------------------------------------------------------------------------------------
    # OTHERS
    # -----------------------------------------------------------------------------------------

    pow_mean = np.sum(T.reshape(-1)* np.abs(constellation)**2)

    tx["const_norm_factor"] = norm_factor   
    tx["constellation"]     = np.expand_dims(constellation,axis = 1)
    tx["N_amps"]            = N_amps                    # number of positive amplitude
    tx["pow_mean"]          = pow_mean                  # mean power of the constellation
    tx["Symb_Probs"]        = Symb_Probs
    tx['Prob_ring']         = Prob_ring
    tx['nu_sc']             = nu_sc

    if rx['mimo'].lower() == "vae":
        tx["amps"]              = torch.tensor(amps)
        tx["prob_amps"]         = torch.tensor(P)       # probabilities of the amplitude levels
    else:
        tx["amps"]              = amps
        tx["prob_amps"]         = P

        
    tx              = misc.sort_dict_by_keys(tx)
    rx              = misc.sort_dict_by_keys(rx)

    return tx,rx


#%%
def set_Nsymbols(tx,fibre,rx):

    # print('########################################################################')
    if fibre['fpol'] != 0:
        rx["NSymbFrame"]    = int(fibre['DeltaThetaC']**2/4*tx['Rs']/fibre['fpol'])   # number of symbols having the same polarisation state
    else:
        rx["NSymbFrame"]    = 10000   # number of symbols having the same polarisation state        
    
    if tx['dnu'] != 0:
        rx["NSymbBatch"]      = int(tx['DeltaPhiC']**2/4*tx['Rs']/tx['dnu'])         # number of symbols having more or less the same phase
    else:
        rx['NSymbBatch']      = 100
    
    tx['NSymbFrame']        = rx["NSymbFrame"]
    tx["Nconv"]             = rx["NSymbFrame"]+tx["Ntaps"]+1
    tx["Nsamp_up"]          = tx["Nsps"]*(tx["Nconv"]-1)+1
    tx["NsampFrame"]        = tx["Nsamp_up"]-(tx["Ntaps"]-1)
    
    flag = 0
    
    if "SymbScale" not in rx:
        flag            = flag + 1
        rx["SymbScale"] = 100
    

    if rx["NSymbBatch"] > rx["NSymbFrame"]:
         sys.exit("ERROR --- NSymbBatch > NSymbFrame")
         
         
         
    # if rx["NSymbBatch"] > 1000:
         # print('WARNING --- NSymbBatch = {}> 1,000'.format(rx['NSymbBatch']))
    
    
    
    if rx['NSymbBatch']%5 != 0:
        flag            = flag +1
        NsymbAdded      = 5-rx['NSymbBatch']%5
        rx['NSymbBatch']  = int(rx['NSymbBatch'] + NsymbAdded)
        
        # print('Nphi = NSymbBatch = {}'.format(rx["NSymbBatch"]))
        
        
        
    if rx['NSymbFrame']%rx["NSymbBatch"] != 0:
        # print('WARNING --- NSymbFrame % NSymbBatch = {} != 0'.format(rx['NSymbFrame']%rx["NSymbBatch"]))  

        flag                = flag +1
        NsymbRemoved        = rx['NSymbFrame']%rx["NSymbBatch"]
        rx['NSymbFrame']    = int(rx['NSymbFrame'] - NsymbRemoved)

        # print('Npol = NsymbFrame = {}'.format(rx["NSymbFrame"]))
        
        
        
    if rx["NSymbFrame"] > 5e4:
        # print('WARNING --- NSymbFrame = {} > 50,000'.format(rx['NSymbFrame']))  
 
        flag                = flag +1        
        rx["NSymbFrame"]    = int(rx["NSymbFrame"]/rx["SymbScale"])
        rx['NSymbBatch']      = int(rx['NSymbBatch']/rx["SymbScale"])
        
        # print('Npol = NsymbFrame = {}'.format(rx["NSymbFrame"]))
        # print('Nphi = NSymbBatch = {}'.format(rx["NSymbBatch"]))
        
        

    if rx['NSymbFrame']%100 != 0:
        # print('WARNING --- NSymbFrame%100 = {} != 0'.format(rx['NSymbFrame']%100))
        
        flag                = flag +1
        NsymbAdded          = 100-rx['NSymbFrame']%100
        rx['NSymbFrame']    = int(rx['NSymbFrame'] + NsymbAdded)
        
        # print('Npol = NsymbFrame = {}'.format(rx["NSymbFrame"]))     
        
        
        
    if rx['NSymbFrame']%rx["NSymbBatch"] != 0:
        # print('WARNING --- NSymbFrame%100 = {} != 0'.format(rx['NSymbFrame']%100))
        
        flag                = flag +1
        NsymbRemoved        = rx['NSymbFrame']%rx["NSymbBatch"]
        rx['NSymbFrame']    = int(rx['NSymbFrame'] - NsymbRemoved)
        
        # print('Npol = NsymbFrame = {}'.format(rx["NSymbFrame"]))
        
    if flag != 0:
        tx['dnu']       = int(tx['DeltaPhiC']**2 * tx['Rs']/4/rx["NSymbBatch"]/rx["SymbScale"])
        fibre['fpol']   = int(fibre['DeltaThetaC']**2 * tx['Rs']/4/rx["NSymbFrame"]/rx["SymbScale"])
        
        print('set-Nsymbols sum up:')
        print('symb scale           = {}'.format(rx["SymbScale"]))
        print('Npol = NsymbFrame    = {}'.format(rx["NSymbFrame"]))
        print('Nphi = NSymbBatch      = {}'.format(rx["NSymbBatch"]))
        print('Npol/Nphi            = {}'.format(rx["NSymbFrame"]/rx["NSymbBatch"]))
        print("fpol                 = {}".format(fibre['fpol']))
        print('dnu                  = {}'.format(tx['dnu']))
    # print('########################################################################')
    # print('\n'*3)
    tx              = misc.sort_dict_by_keys(tx)
    fibre           = misc.sort_dict_by_keys(fibre)
    rx              = misc.sort_dict_by_keys(rx)
    return tx,fibre,rx



#%%

def shaping_filter(tx): 

    if "shaping_filter" not in tx:
        tx['shaping_filter'] = "dirac"

    if tx['shaping_filter'].lower() == 'dirac':
        h               = np.zeros(tx['Ntaps'])
        i               = round(tx['Ntaps']/2)
        h[i]            = 1

    elif tx['shaping_filter'].lower() == 'random':
        np.random.seed(0)
        h       = np.random.rand(tx['Ntaps'])/3
        i       = round(tx['Ntaps']/2)
        h[i]    = 0
        energy  = h.dot(h)
        h[i]    = np.sqrt(1-energy)

    else:

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
        h       = h/np.linalg.norm(h)

    tx['hmatrix']       = np.zeros((4,tx['Ntaps']))
    tx['hmatrix'][0]    = h
    tx['hmatrix'][3]    = h

    # gen.plot_fir(tx,tx['hmatrix'])

    tx      = misc.sort_dict_by_keys(tx)

    return tx

#%%
def transmitter(tx,rx):


    tx          = data_generation(tx,rx)
    tx          = shaping_filter(tx)
    tx          = data_shaping(tx)
    
        
    if rx["Frame"] >= rx["FrameChannel"]:

        tx      = txhw.load_ase(tx,rx)
        tx      = txhw.load_phase_noise(tx,rx)
    

        
    tx              = misc.sort_dict_by_keys(tx)
    return tx






