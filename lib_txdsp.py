# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.2.3
#   Date            : 2024-06-27
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2024-03-04)  - creation
#   1.0.3 (2024-04-03)  - cleaning
#   1.0.4 (2024-05-27)  - get_constellation: inspiration from [C2]
#   1.1.0 (2024-06-04)  - get_constellation: adding adjective (pilots,)
#                       - set_Nsymbols: displaying net symbols added/removed
#                       - [NEW] pilot_generation
#                       - transmitter: adding pilots management
#   1.1.1 (2024-06-06)  - set_Nsymbols, cleaning (Ntaps, Nconv)
#   1.2.0 (2024-06-07)  - pilot_generation: scaling to data power
#                       - [NEW] pilot_insertion
#                       - transmitter: including pilot_insertion
#   1.2.1 (2024-06-11)  - pilot_generation: including batch/frame wise
#   1.2.2 (2024-06-14)  - pilot_generation/insertion, robustness + cazac + data
#   1.2.3 (2024-06-27)  - transmitter, pilots_generation/insertion:
#                                 varargins for checking steps
#                         set_Nsymbols: number of pilots
#                         transmitter: flag phase noise
#
# ----- MAIN IDEA -----
#   Library for Digital Signal Processing at the Transmitter side in (optical)
#   telecommunications
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   Authors             : Sterenn Guerrier
#   Title               : High bandwidth detection of mechanical stress in
#                         optical fibre using coherent detection of Rayleigh
#                         scattering
#   Journal/Editor      : Télécom Paris, Institut Polytechnique de Paris
#   Volume - N°         : PhD thesis
#   Date                : 2022-02-03
#   DOI/ISBN            : NNT: 2022IPPAT004
#   Pages               : 28, eq 1.10
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
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import lib_misc as misc
import lib_general as gen
import lib_txhw as txhw
import lib_matlab as mb

pi = np.pi

from lib_matlab import clc
from lib_misc import KEYS as keys
from lib_maths import get_power as power



#%% ===========================================================================
# --- CONTENTS ---
# =============================================================================
# - data_generation                 --- called in : transmitter
# - data_shaping                    --- called in : transmitter
# - get_constellation               --- called in : processing.init_processing
# - pilot_generation        (1.1.0) --- called in : transmitter
# - pilot_insertion         (1.2.0) --- called in : transmitter
# - set_Nsymbols                    --- called in : main.process_data
# - shaping_filter                  --- called in : transmitter
# - transmitter                     --- called in : processing.processing
# =============================================================================



#%%
def transmitter(tx,rx):

    tx          = shaping_filter(tx)
    tx          = data_generation(tx,rx)
    
    for what_pilots_k in range(len(tx['pilots_info'])):
        
        tx      = pilot_generation(tx, rx, what_pilots_k)#,"show")             # uncomment to check
        tx,rx   = pilot_insertion(tx, rx, what_pilots_k)#,'show')              # uncomment to check

    tx          = data_shaping(tx)#,rx)                                        # uncomment to check

        

    # let the vae learn the rrc filter
    if rx["Frame"] >= rx["FrameChannel"]:
        tx      = txhw.load_ase(tx,rx)

        if tx['flag_phase_noise']:
            tx      = txhw.load_phase_noise(tx,rx)#'pn const','pn time trace')     # uncomment to check

        # gen.plot_constellations(tx['sig_real'],\
        # title = f"tx, frame = {rx['Frame']},\
        # pn = {round(180/pi*np.mean(tx['PhaseNoise'][0,:,rx['Frame']]),3)}")
    
    tx          = misc.sort_dict_by_keys(tx)
    return tx





#%%


#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================

#%%
def data_generation(tx,rx,*what_pilots):

    # If P == uniform, the plt.plot(data[0]) will be looking
    # like filled rectangle for instance:
    #
    #     p = (np.ones((1,len(amps)))*1/len(amps)).reshape(-1)
    #
    # DATA = 
    #   [
    #       data array with pol-H channel-I, shape = (1,NSymbConv)
    #       data array with pol-H channel-Q, shape = (1,NSymbConv)
    #       data array with pol-V channel-I, shape = (1,NSymbConv)
    #       data array with pol-V channel-Q, shape = (1,NSymbConv)
    #   ] =
    #   [
    #       sHI[0],sHI[1],...,sHI[NSymbConv-1]
    #       sHQ[0],sHQ[1],...,sHQ[NSymbConv-1]
    #       sVI[0],sVI[1],...,sVI[NSymbConv-1]
    #       sVQ[0],sVQ[1],...,sVQ[NSymbConv-1]
    #   ]
    # DATA_I = 
    #   [
    #       sHI[0],sHI[1],...,sHI[NSymbConv-1]
    #       sVI[0],sVI[1],...,sVI[NSymbConv-1]
    #   ]

    #1 data arrays parameters
    # # NSymbConv = NSymbFrame + some extra symbols necessary for
    # # edges management
    # tx["NSymbConv"] = rx["NSymbFrame"]+tx['NsampTaps']+1
    # tx["Nsamp_up"]  = tx["Nsps"]*(tx["NSymbConv"]-1)+1

    #2 data generation
    # draw randomly amplitude values from AMPS using the law 
    # of probability used in the Probabilistic Constellation
    # Shaping.
    
    data            = np.random.default_rng().choice(tx["amps"],
                                            (2*tx["Npolars"],tx["NSymbConv"]),
                                            p=np.round(tx["prob_amps"],5))
    data_I          = data[0::tx["Npolars"],:]
    data_Q          = data[1::tx["Npolars"],:]

    # sps-upsampled signal by zero-insertion
    # louis, do not remove the type definition
    tx["sig_cplx_up"]   = np.zeros((tx["Npolars"],tx["Nsamp_up"]),\
                                   dtype=np.complex64)
    tx["sig_cplx_up"][:,::tx["Nsps"]]   = data_I + 1j*data_Q
    
    pow_I           = np.mean(np.abs(data_I)**2)
    pow_Q           = np.mean(np.abs(data_Q)**2)
    tx['sig_power'] = pow_I+pow_Q

    mask_start      = int(tx['NsampTaps']/2)
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
def data_shaping(tx,*varargin):

    # # The "valid" mode for convolution already shrinks the length
    # tx["NsampFrame"]  = tx["Nsamp_up"]-(tx['NsampTaps']-1)
    # 0 == pol H ------- 1 == pol V

    h_pulse             = tx['hmatrix'][0]
    tx["sig_real"]      = np.zeros((tx["Npolars"]*2,tx["NsampFrame"]),\
                                   dtype=np.float32)

    tmp                 = np.convolve(tx["sig_cplx_up"][0,:], h_pulse,\
                                  mode = 'valid')
        
    tx["sig_real"][0]   = np.real(tmp)
    tx["sig_real"][1]   = np.imag(tmp)

    tmp                 = np.convolve(tx["sig_cplx_up"][1,:], h_pulse,\
                                  mode = 'valid')
        
    tx["sig_real"][2]   = np.real(tmp)
    tx["sig_real"][3]   = np.imag(tmp)

    del tx["sig_cplx_up"], tmp



    # ---------------------------------------------------------------- to check
    if len(varargin) > 0 and varargin is not None:
        rx = varargin[0]
        # cf. data_shaping, convolution with filter, mode valid
        # 
        # y         = conv(x,h,'valid) of len(x) = N, len(h) = M
        # len(y)    = N-M+1
        #
        # formula: mode full    y[n] = sum_(k=0^{n})   x[k]   . h[n-k]
        # formula: mode valid   y[n] = sum_(k=0^{M-1}) x[k+n] . h[n-k]

        x1 = tx['Nsamp_pilots_cpr']-tx['NsampTaps']+1
        for k in range(5):
            kstart  = k*tx['NsampBatch']
            kend    = kstart + tx['Nsamp_pilots_cpr']
            plt.figure()
            
            plt.subplot(2,2,1)
            plt.plot(tx["sig_real"][0,kstart:kend])
            plt.plot([x1,x1],[-1,1])
            plt.title("HI")
            plt.ylim([-1,1])
    
            plt.subplot(2,2,2)
            plt.plot(tx["sig_real"][1,kstart:kend])
            plt.plot([x1,x1],[-1,1])
            plt.title("HQ")
            plt.ylim([-1,1])
    
            plt.subplot(2,2,3)
            plt.plot(tx["sig_real"][2,kstart:kend])
            plt.plot([x1,x1],[-1,1])
            plt.title("VI")
            plt.ylim([-1,1])
    
            plt.subplot(2,2,4)
            plt.plot(tx["sig_real"][3,kstart:kend])
            plt.plot([x1,x1],[-1,1])
            plt.title("VQ")
            plt.ylim([-1,1])
    
            plt.suptitle(f"data shaping {rx['Frame']} - {kstart}-{kend}")
            plt.show()
    # ---------------------------------------------------------------- to check
            
    tx                  = misc.sort_dict_by_keys(tx)

    return tx

#%%
def get_constellation(tx,rx,*what_pilots):
    
############################################################################### 
######################### CONSTELLATION CONSTRUCTION ##########################
###############################################################################
    
    # parameters definition of each constellation
    constellations = {}
    params = {
        'BPSK'  :   (2, [-1, 1]),
        '4QAM'  :   (2, [-1, 1]),
        '16QAM' :   (4, [-3, -1, 1, 3]),
        '64QAM' :   (8, [-7, -5, -3, -1, 1, 3, 5, 7]),
    }
    
    if len(what_pilots) != 0:
        pilots_function = 'pilots_' + what_pilots[0][0].lower()
        pilots_method   = what_pilots[0][1].lower()
        pilots_mod      = what_pilots[0][4]

        
    if len(what_pilots) == 0:
        Nradii = {
            'BPSK'  :   (1, [2]),
            '4QAM'  :   (1, [4]),
            '16QAM' :   (3, [4, 8, 4]),
            '64QAM' :   (9, [4, 8, 4, 8, 8, 12, 8, 8, 4])
        }
    
    # creation of available constellations
    if 'psk' not in tx['mod'].lower():
        for modfor, (side, amplitudes) in params.items():
            points = []
            for i in range(side):
                for j in range(side):
                    points.append(amplitudes[i] + 1j * amplitudes[j])
            constellations[modfor]  = np.array(points)
    elif tx['mod'].lower() == 'bpsk':
        constellations[modfor] = np.array([1j,-1j])

    # norm factor: sqrt(2*(M-1)/3)
    # 4qam: sqrt(2) -- 16 qam: sqrt(10) -- 64qam: sqrt(42) -- 256qam: sqrt(170)
    if len(what_pilots) == 0:
        tx["const_affixes"]     = constellations[tx["mod"]]
    else:
        # if "cazac" not in pilots_method:
        if "cazac" not in pilots_method and "data" not in pilots_method:
            tx["const_affixes_{}".format(pilots_function)] = constellations[pilots_mod]
        
    if len(what_pilots) == 0:
        rings                   = np.unique(abs(tx["const_affixes"]))
        Prob_symb_tmp           = np.exp(-tx['nu']*rings**2)
        Prob_ring_tmp           = Prob_symb_tmp*Nradii[tx['mod']][1]
        Prob_symb               = Prob_symb_tmp/sum(Prob_ring_tmp)
        Prob_ring               = Prob_symb*Nradii[tx['mod']][1]
        
        Symb_Probs_tmp          = np.exp(-tx['nu']*abs(tx["const_affixes"]))
        Symb_Probs              = Symb_Probs_tmp/sum(Symb_Probs_tmp)

    if len(what_pilots) == 0:
        norm_factor             = np.sqrt(np.mean(abs(tx["const_affixes"])**2))
        constellation           = tx["const_affixes"]/norm_factor
    else:
        # if "cazac" not in pilots_method:
        if "cazac" not in pilots_method and "data" not in pilots_method:
            norm_factor_pilots      = np.sqrt(np.mean(abs(\
                                  tx["const_affixes_{}".format(pilots_function)]**2)))
                
            constellation_pilots    = tx["const_affixes_{}".format(pilots_function)]
            constellation_pilots    /= norm_factor_pilots
        
    # misc.plot_complex_constellation(constellation)
    
###############################################################################
############################### PROBABILITY LAW ###############################
###############################################################################

    # extract real amplitudes

    if  len(what_pilots) == 0:
        # number of ASK levels
        N_amps          = int(np.sqrt(len(constellation.real)))
        
        # amplitude levels
        amps            = constellation.real[::N_amps]
        sc              = min(abs(amps))
        
        # scaling factor for having lowest level equal 1
        nu_sc           = tx['nu']/(sc**2)
    
        # probabilities of the amlitude levels
        P               = np.exp(-tx["nu"]*np.abs(amps/sc)**2)
        P               = P / sum(P)  
        P               = np.expand_dims(P, axis=0)
        T               = np.dot(P.T,P)
        P               = P.reshape(-1)
        
    else:
        # if "cazac" not in pilots_method:
        if "cazac" not in pilots_method and "data" not in pilots_method:
            N_amps_pilots   = int(np.sqrt(len(constellation_pilots.real)))
            amps_pilots     = constellation_pilots.real[::N_amps_pilots]

    # mb.imagesc(np.log10(T)) # plot the probability matrix
###############################################################################
################################### SAVING ####################################
###############################################################################

    if  len(what_pilots) == 0:
        pow_mean = np.sum(T.reshape(-1)* np.abs(constellation)**2)
    else:
        # if "cazac" not in pilots_method:
        if "cazac" not in pilots_method and "data" not in pilots_method:
            pow_mean_pilots = np.sum(np.abs(constellation_pilots)**2)

    if  len(what_pilots) == 0:
        tx["const_norm_factor"] = norm_factor
        tx["constellation"]     = np.expand_dims(constellation,axis = 1)
        
        # number of positive amplitude
        tx["N_amps"]            = N_amps
        
        # mean power of the constellation
        tx["pow_mean"]          = pow_mean
        tx["Symb_Probs"]        = Symb_Probs
        tx['Prob_ring']         = Prob_ring
        tx['nu_sc']             = nu_sc
    
        if rx['mimo'].lower() == "vae":
            tx["amps"]          = torch.tensor(amps)
            
            # probabilities of the amplitude levels
            tx["prob_amps"]     = torch.tensor(P)
        else:
            tx["amps"]          = amps
            tx["prob_amps"]     = P
    else:
        # if "cazac" not in pilots_method:
        if "cazac" not in pilots_method and "data" not in pilots_method:
            tx["const_norm_factor_{}".format(pilots_function)]= norm_factor_pilots
            tx["constellation_{}".format(pilots_function)]    = np.expand_dims(constellation_pilots,axis = 1)
            tx["N_amps_{}".format(pilots_function)]           = N_amps_pilots
            tx["pow_mean_{}".format(pilots_function)]         = pow_mean_pilots
        
            if rx['mimo'].lower() == "vae":
                tx["amps_{}".format(pilots_function)]         = torch.tensor(amps_pilots)
            else:
                tx["amps_{}".format(pilots_function)]         = amps_pilots

        
    tx  = misc.sort_dict_by_keys(tx)
    rx  = misc.sort_dict_by_keys(rx)

    return tx,rx


#%%
# what_pilots[k] = pilots_info
#
# 0 = {cpr, synchro_once, synchro_frame} === pilots locations
#   cpr             : batch-wise
#   synchro_once    : first batch of first frame is used, once for all
#   synchro_frame   : first batch of each frame
#
# 1 = {rand, file, custom, cazac, data, ...} pilots selection method
#   rand            : uniformly drawn symbols
#   file            : upload from file col1 = amps_I, col2 = amps_Q,
#                                      col3 = phis_I, col4 = phis_Q
#   custom          : manually write the selection method
#   data            : use data as pilots
#   cazac           : CAZAC sequences (Constant Amplitude Zero-Autocorrelation
#                     Code)
#
# 2 = {fixed, different} =================== pilots changes
#   fixed           : identical pilots each time they are generated
#   different       : different "   " "   " "   " "   " "   " "   "
#
# 3 = {same, polwise} ====================== same for both polarisations or not
#   same            : identical for each signal polarisation
#   polwise         : different "   " "   " "   " "   " "   "
# 
# 4 = {4, 16, 64}QAM ======================= modulation format used
# 5 = {>0} ================================= percentage of pilots if not cazac
#                                            number of cazac symbol otherwise
#
# 6 = {>0} ================================= number pilots/batch if not cazac

def pilot_generation(tx,rx,what_pilots_k,*varargin):

###############################################################################
############################### Sub functions #################################
###############################################################################

    what_pilots = tx['pilots_info'][what_pilots_k]
    
    def set_params(what_pilots):
        
        pilots_function = 'pilots_' + what_pilots[0].lower()
        pilots_changes  = what_pilots[2].lower()
        pilots_polwise  = what_pilots[3].lower()
        pilots_mod      = what_pilots[4].lower()
        
        # ------------------------------------------------------------- FLAG DO
        if (pilots_changes == 'same')\
            and '{}_flag_all_same'.format(pilots_function) in tx:
            flag_do = 0
                
        elif (pilots_changes == 'same')\
            and '{}_flag_all_same'.format(pilots_function) not in tx:    
            flag_do = 1
            
        elif ('once' in pilots_function)\
            and '{}_flag_all_same'.format(pilots_function) in tx:
            flag_do = 0
                
        elif pilots_changes != 'same':
            flag_do = 1
            
        # ----------------------------------------------------- FLAG REDO (POL)
        if flag_do and "synchro" not in pilots_function:
            if pilots_polwise == "same":
                N_redo_pol = 1
            else:
                N_redo_pol = 2
                
            if pilots_changes == "same":
                N_redo = 1
            else:
                N_redo = rx['NBatchFrame']
                
        if flag_do and "synchro" in pilots_function:

            if 'pol' not in what_pilots[3]:
                N_redo_pol = 1
            else:
                N_redo_pol = 2

            if (pilots_changes == "same") or ('once' in pilots_function):
                N_redo = 1
            else:
                N_redo = rx['Nframes']
                
        if not flag_do:
            N_redo      = 0
            N_redo_pol  = 0
            
        if "cazac" in what_pilots:
            N_redo      = 1
            N_redo_pol  = 1
            
            
        # if flag_do:
        if (tx['mimo'].lower() == "vae")\
            and ('synchro' in pilots_function.lower())\
            and (what_pilots[1].lower() == 'data'):
            shape   = (N_redo_pol,N_redo,tx['NSymbBatch'])

        elif what_pilots[2].lower() == 'cazac':
            shape   = (N_redo_pol,N_redo,what_pilots[-1])

        else:
            shape   = (N_redo_pol,N_redo,tx['NSymb_{}'.\
                                    format(pilots_function)])

        return flag_do, N_redo, N_redo_pol, pilots_function, pilots_mod, shape

    # ----------------------------------------------------------------------- #
    
    def gen_cazac(tx):
        
        NSymbCazac  = tx['NSymb_{}'.format(pilots_function)]
        tmp1        = 2*pi/np.sqrt(NSymbCazac)
        tmpN        = np.linspace(0, NSymbCazac-1,NSymbCazac)
        tmp2        = 1+np.mod(tmpN,np.sqrt(NSymbCazac))
        tmp3        = 1+np.floor(tmpN/np.sqrt(NSymbCazac))
        
        pilots_I    = np.real(np.exp(1j*tmp1*tmp2*tmp3))
        pilots_Q    = np.imag(np.exp(1j*tmp1*tmp2*tmp3)) # (64,)
        
        return pilots_I,pilots_Q
    # ----------------------------------------------------------------------- #
        
    def gen_custom(tx):
        
        amps_I      = np.ones(tx['NSymb_{}'.format(pilots_function)])
        amps_Q      = np.ones(tx['NSymb_{}'.format(pilots_function)])
        phis_I      = np.ones(tx['NSymb_{}'.format(pilots_function)])*pi/4
        phis_Q      = np.ones(tx['NSymb_{}'.format(pilots_function)])*pi/4
        
        pilots_I    = np.real(amps_I*np.exp(1j*phis_I)) # (5,)
        pilots_Q    = np.imag(amps_Q*np.exp(1j*phis_Q))
        
        return pilots_I,pilots_Q
    # ----------------------------------------------------------------------- #
    
    def gen_from_data(tx,frame,pol):
        
        if tx['mimo'].lower() == "vae":
            index_end   = tx['NSymbBatch']
        else:
            index_end   = tx['NSymb_{}'.format(pilots_function)]
            
        pilots_I    = tx['Symb_real'][0][pol][0:index_end]
        pilots_Q    = tx['Symb_real'][1][pol][0:index_end]
        
        return pilots_I,pilots_Q
    # ----------------------------------------------------------------------- #
        
    def gen_from_file(tx):
        
        file        = open(tx['pilots_explicit'], 'r')
        pilots      = file.read()
        file.close()
        
        amps_I      = pilots[:,0]
        amps_Q      = pilots[:,1]
        phis_I      = pilots[:,2]
        phis_Q      = pilots[:,3]
        
        pilots_I    = np.real(amps_I*np.exp(1j*phis_I))
        pilots_Q    = np.imag(amps_Q*np.exp(1j*phis_Q))
        
        return pilots_I,pilots_Q
    # ----------------------------------------------------------------------- #
        
    def gen_rand(tx):

        pilots_Q = np.random.default_rng().choice(\
                        tx["amps_{}".format(pilots_function)],
                        tx["NSymb_{}".format(pilots_function)])
            
        if 'psk' not in pilots_mod:
            pilots_I = np.random.default_rng().choice(\
                            tx["amps_{}".format(pilots_function)],
                            tx["NSymb_{}".format(pilots_function)])
        else:
            pilots_I = np.zeros(pilots_Q.shape)

        return pilots_I,pilots_Q
    # ----------------------------------------------------------------------- #
    
    def pilots_processing(tx,pilots_I,pilots_Q,what_pilots_k):
        
        what_pilots             = tx['pilots_info'][what_pilots_k]
        pilots_I                = np.round(pilots_I,4)
        pilots_Q                = np.round(pilots_Q,4)

        pow_pilots_I            = np.mean(np.abs(pilots_I)**2)
        pow_pilots_Q            = np.mean(np.abs(pilots_Q)**2)
        pow_pilots_tot          = pow_pilots_I+pow_pilots_Q
        tx['sig_pilots_ratio']  = np.sqrt(tx['sig_power']/pow_pilots_tot)

        pilots_cplx  = (pilots_I + 1j*pilots_Q)*tx['sig_pilots_ratio']
        # pilots_cplx             = (pilots_I + 1j*pilots_Q)*10
        
        if (what_pilots[3].lower() == "same") or ("cazac" in what_pilots):
            pilots_cplx = mb.repmat(pilots_cplx, (tx['Npolars'],1,1))

        tx["{}_cplx_up".format(pilots_function)]     = \
        np.zeros((tx["Npolars"],N_redo,tx["Nsamp_{}".format(pilots_function)]),
                 dtype=np.complex64)
            
        tx["{}_cplx_up".format(pilots_function)][:,:,::tx["Nsps"]]= pilots_cplx

        tmpH = tx['{}_cplx_up'.format(pilots_function)][0]
        tmpV = tx['{}_cplx_up'.format(pilots_function)][1]
            
        # if same pilots on both polar all the symbols should be equal
        # otherwise they must be different 
        if N_redo_pol == 1:
            assert np.sum(tmpH == tmpV) == tx["Nsamp_{}".format(pilots_function)]*N_redo
        else:
            assert np.sum(tmpH == tmpV) != tx["Nsamp_{}".format(pilots_function)]*N_redo
        
        if tx['pilots_info'][what_pilots_k][-1] == 0:
            tx['pilots_info'][what_pilots_k][-1] = int(tx["Nsamp_{}".format(pilots_function)]/tx['Nsps'])
        
        return tx
    # ----------------------------------------------------------------------- #
        

#%%
###############################################################################
################################# generation ##################################
###############################################################################

    if rx['mode'].lower() != "blind":
        
        flag_do, N_redo, N_redo_pol, pilots_function, pilots_mod, shape\
            = set_params(what_pilots)

        if flag_do:

            pilots_I    = np.zeros(shape)
            pilots_Q    = np.zeros(shape)

            for k in range(N_redo):
                for j in range(N_redo_pol):
    
                    if what_pilots[1] == 'cazac':
                        pilots_I[:,k], pilots_Q[:,k] = gen_cazac(tx)

                    elif what_pilots[1] == 'custom':
                        pilots_I[j,k], pilots_Q[j,k] = gen_custom(tx)
                        
                    elif 'data' in what_pilots[1]: # if synchro
                        pilots_I[j,k], pilots_Q[j,k] = gen_from_data(tx,k,j)
                        
                    elif 'file' in what_pilots[1]:
                        pilots_I[j,k], pilots_Q[j,k] = gen_from_file(tx)
                        
                    elif what_pilots[1] == "rand":
                        pilots_I[j,k], pilots_Q[j,k] = gen_rand(tx)


            # scaling the pilots power, upsampling and saving into tx dict
            tx  = pilots_processing(tx, pilots_I, pilots_Q,what_pilots_k)
            tx  = misc.sort_dict_by_keys(tx)
        

    # for checking
    if len(varargin) > 0 and varargin is not None:
        tmpH = tx['{}_cplx_up'.format(pilots_function)][0].squeeze()
        tmpV = tx['{}_cplx_up'.format(pilots_function)][1].squeeze()
        
        # cf. data_shaping, convolution with filter, mode valid
        # conv(x1,x2,'valid) of len N1-N2+1
        x1 = tx['Nsamp_pilots_cpr']-tx['NsampTaps']+1
        plt.figure()
        
        plt.subplot(2,2,1)
        plt.plot(np.real(tmpH))
        plt.plot([x1,x1],[-1,1])
        plt.title("HI")
        plt.ylim([-1,1])

        plt.subplot(2,2,2)
        plt.plot(np.imag(tmpH))
        plt.plot([x1,x1],[-1,1])
        plt.title("HQ")
        plt.ylim([-1,1])

        plt.subplot(2,2,3)
        plt.plot(np.real(tmpV))
        plt.plot([x1,x1],[-1,1])
        plt.title("VI")
        plt.ylim([-1,1])

        plt.subplot(2,2,4)
        plt.plot(np.imag(tmpV))
        plt.plot([x1,x1],[-1,1])
        plt.title("VQ")
        plt.ylim([-1,1])

        plt.suptitle("pilot_generation, frame = {}".format(rx['Frame']))
        plt.show()
        # gen.plot_constellations(tx['{}_cplx_up'.format(pilots_function)],polar='both',sps=2)
            
    return tx


#%%
# what_pilots[k] = pilots_info
#
# 0 = {cpr, synchro_once, synchro_frame} ======== pilots locations
# 1 = {rand, file, custom, cazac, data, ...} ==== pilots selection method
# 2 = {fixed, different} ======================== pilots changes
# 3 = {same, polwise} =========================== same for both polarisations or not
# 4 = {4, 16, 64}QAM ============================ modulation format used
# 5 = {>0} ====================================== percentage of pilots if not cazac
#                                                   number of cazac symbol otherwise
# 6 = {>0} ====================================== number of pilots per batch if not cazac
 
def pilot_insertion(tx,rx,what_pilots_k,*varargin):    

    what_pilots     = tx['pilots_info'][what_pilots_k]
    
    if rx['mode'].lower() != "blind":
        pilots_function     = what_pilots[0].lower()
        pilots_changes      = what_pilots[2].lower()
        
        if ('once' in pilots_function) or ('same' in pilots_changes):
            tx['pilots_{}_flag_all_same'.format(pilots_function)] = 1
            
        if "synchro" in pilots_function:
            if '{}_flag_all_same'.format(pilots_function) not in tx:
                index_start     = 0
                index_end       = index_start + tx['Nsamp_pilots_{}'.format(pilots_function)]
                    
                if pilots_function == "synchro_once":
                    tx['sig_cplx_up'][:,index_start:index_end] =\
                        tx['pilots_{}_cplx_up'.format(pilots_function)].squeeze()
                
                elif pilots_function == "synchro":
                    if "cazac" in what_pilots:
                        tx['sig_cplx_up'][:,index_start:index_end] =\
                            tx['pilots_{}_cplx_up'.format(pilots_function)].squeeze()
                    else:
                        tx['sig_cplx_up'][:,index_start:index_end] =\
                            tx['pilots_{}_cplx_up'.format(pilots_function)][:,rx['Frame'],:]
                    
                else:
                    tx['sig_cplx_up'][:,index_start:index_end] =\
                        tx['pilots_{}_cplx_up'.format(pilots_function)][:,rx['Frame']]
    # ----------------------------------------------------------------------- #
    
        else:
            for k in range(rx['NBatchFrame']):
                
                index_start = 0+k*rx['NsampBatch']
                index_end   = index_start + tx['Nsamp_pilots_{}'.format(pilots_function)]
                
                if pilots_changes == "same":
                    pilots      = tx['pilots_{}_cplx_up'.format(pilots_function)].squeeze()
                else:
                    if "cazac" in what_pilots:
                        pilots  = tx['pilots_{}_cplx_up'.format(pilots_function)].squeeze()
                    else:
                        pilots  = tx['pilots_{}_cplx_up'.format(pilots_function)][:,k,:]
                    
                tx['sig_cplx_up'][:,index_start:index_end] = pilots
                
        tx['Symb_{}_cplx'.format(pilots_function)] =\
            tx['pilots_{}_cplx_up'.format(pilots_function)][:,:,::tx['Nsps']]
        # gen.plot_constellations(tx['Symb_{}_cplx'.format(pilots_function)])

        if ('once' in pilots_function) or ('same' in pilots_changes):
            tx['{}_flag_all_same'.format(pilots_function)] = 1
        
        
        
    # for checking
    if len(varargin) > 0 and varargin is not None:
        tmpH = tx['Symb_{}_cplx'.format(pilots_function)][0].squeeze()
        tmpV = tx['Symb_{}_cplx'.format(pilots_function)][1].squeeze()
        
        plt.figure()
        
        plt.subplot(2,2,1)
        plt.plot(np.real(tmpH))
        plt.title("HI")
        plt.ylim([-1,1])

        plt.subplot(2,2,2)
        plt.plot(np.imag(tmpH))
        plt.title("HQ")
        plt.ylim([-1,1])

        plt.subplot(2,2,3)
        plt.plot(np.real(tmpV))
        plt.title("VI")
        plt.ylim([-1,1])

        plt.subplot(2,2,4)
        plt.plot(np.imag(tmpV))
        plt.title("VQ")
        plt.ylim([-1,1])

        plt.suptitle("pilots insertion {}".format(rx['Frame']))
        plt.show()

    return tx,rx

#%%
def set_Nsymbols(tx,fibre,rx):

    # number of symbols having the same polarisation state
    # if fibre['fpol'] != 0:
    #     rx["NSymbFrame"]    = int(fibre['DeltaThetaC']**2/4*tx['Rs']/fibre['fpol'])
    # else:
    #     rx["NSymbFrame"]    = 10000
    
    # # number of symbols having more or less the same phase
    # if tx['dnu'] != 0:
    #     rx["NSymbBatch"]    = int(tx['DeltaPhiC']**2/4*tx['Rs']/tx['dnu'])
    # else:
    #     rx['NSymbBatch']    = 100
    

        
    tx['NSymbFrame']        = rx["NSymbFrame"]
    tx["NSymbConv"]         = rx["NSymbFrame"]+tx['NSymbTaps']+1
    tx["Nsamp_up"]          = tx["Nsps"]*(tx["NSymbConv"]-1)+1
    tx['NsampTaps']         = tx["Nsps"]*tx["NSymbTaps"]-1      # length of FIR filter
    tx["NsampFrame"]        = tx["Nsamp_up"]-(tx['NsampTaps']-1)

###############################################################################
######### adjustment of the number of symbols for phase/pol dynamics ##########
###############################################################################
    
    flag                    = 0
    Nsymb_added_net         = 0
    
    if "SymbScale" not in rx:
        flag                = flag + 1
        rx["SymbScale"]     = 100

    if rx["NSymbBatch"] > rx["NSymbFrame"]:
         print("ERROR --- NSymbBatch > NSymbFrame")
         exit
    
    if rx['NSymbBatch']%5 != 0:
        flag                = flag +1
        NsymbAdded          = 5-rx['NSymbBatch']%5
        rx['NSymbBatch']    = int(rx['NSymbBatch'] + NsymbAdded)
        Nsymb_added_net     +=  NsymbAdded
        
    if rx['NSymbFrame']%rx["NSymbBatch"] != 0:
        flag                = flag +1
        NsymbRemoved        = rx['NSymbFrame']%rx["NSymbBatch"]
        rx['NSymbFrame']    = int(rx['NSymbFrame'] - NsymbRemoved)
        Nsymb_added_net     -=  NsymbRemoved

    if rx["NSymbFrame"] > 5e4:
        flag                = flag +1        
        rx["NSymbFrame"]    = int(rx["NSymbFrame"]/rx["SymbScale"])
        rx['NSymbBatch']    = int(rx['NSymbBatch']/rx["SymbScale"])
        Nsymb_added_net     /= rx["SymbScale"]

    if rx['NSymbFrame']%100 != 0:
        flag                = flag +1
        NsymbAdded          = 100-rx['NSymbFrame']%100
        rx['NSymbFrame']    = int(rx['NSymbFrame'] + NsymbAdded)
        Nsymb_added_net     += NsymbAdded
        
    if rx['NSymbFrame']%rx["NSymbBatch"] != 0:
        flag                = flag +1
        NsymbRemoved        = rx['NSymbFrame']%rx["NSymbBatch"]
        rx['NSymbFrame']    = int(rx['NSymbFrame'] - NsymbRemoved)
        Nsymb_added_net     -= NsymbRemoved
        
#%%
###############################################################################
############################# pilots management ###############################
###############################################################################
    if rx['mode'].lower() != "blind":
        NSymbs_pilots = 0
        for k in range(len(tx['pilots_info'])):

            what_pilots     = tx['pilots_info'][k]
            pilots_function = 'pilots_' + what_pilots[0]
            pilots_method   = what_pilots[1]
            
            if 'cazac' in pilots_method:
                NSymbCazac      = what_pilots[-1]
                tmp1            = np.log(NSymbCazac)/np.log(4)                
                assert abs(tmp1-int(tmp1))<1e-4
                # requirement : NSymCazac = 4^(integer)
                tx['NSymb_{}'.format(pilots_function)] = NSymbCazac
                
            elif 'data' in pilots_method:
                if 'once' in pilots_function:
                    tx['NSymb_{}'.format(pilots_function)] = rx['NSymbBatch']
                else:
                    tx['NSymb_{}'.format(pilots_function)] = what_pilots[-1]
                
            else:
                percentage                      = what_pilots[-2]/100
                tx['NSymb_{}'.format(pilots_function)] = int(round(percentage*rx['NSymbBatch']))
                tx['pilots_info'][k][-1]        = tx['NSymb_{}'.format(pilots_function)]
                
            tx['Nsamp_{}'.format(pilots_function)]     = tx['NSymb_{}'.format(pilots_function)]*tx['Nsps']
            rx['NSymb_{}'.format(pilots_function)]     = tx['NSymb_{}'.format(pilots_function)]
            rx['Nsamp_{}'.format(pilots_function)]     = tx['Nsamp_{}'.format(pilots_function)]
            
            NSymbs_pilots += tx['NSymb_{}'.format(pilots_function)]
            

            
            
        # rx['NSymb_data_Frame']              = rx['NSymbBatch']-tx['NSymb_{}'.format(pilots_function)]
        # rx['NSymb_pilots_tot_Batch']        = NSymbs_pilots
        # rx['NSymb_overhead_percent']        = round(rx['NSymb_pilots_tot_Batch']/rx['NSymb_data_Batch']*100,2)
        # rx['Rs_eff']                        = round(rx['NSymb_data_Batch']/rx['NSymbBatch']*tx['Rs']*1e-9,2)
            
#%%
###############################################################################
############################ updating the numbers #############################
###############################################################################
    
    tx['NSymbFrame']        = rx["NSymbFrame"]
    tx["NSymbConv"]         = rx["NSymbFrame"]+tx['NSymbTaps']+1
    tx["Nsamp_up"]          = tx["Nsps"]*(tx["NSymbConv"]-1)+1
    tx['NsampTaps']         = tx["Nsps"]*tx["NSymbTaps"]-1      # length of FIR filter
    tx["NsampFrame"]        = tx["Nsamp_up"]-(tx['NsampTaps']-1)

#%%
###############################################################################
########################### miscellaneous #####################################
###############################################################################
    
    tx['NSymbTot']          = tx['NSymbFrame']*rx['Nframes']
    tx['NsampTot']          = tx['NsampFrame']*rx['Nframes']
    rx["NsampBatch"]        = rx['NSymbBatch']*tx['Nsps']    
    

    # number of symbols cut off to prevent edge effects of convolution
    rx["NSymbCut"]          = 10
    rx["NSymbCut_tot"]      = 2*rx['NSymbCut']+1
    
    rx['NSymbEq']           = rx["NSymbFrame"]
    if rx['mimo'].lower() != "vae":
        rx['NSymbEq']       -= rx['NSymbCut_tot']-1
        
        


###############################################################################
############################# displaying results ##############################
###############################################################################
    
    # if flag != 0:
    # tx['dnu']           = int(tx['DeltaPhiC']**2 * tx['Rs']/4/rx["NSymbBatch"]/rx["SymbScale"])
    # fibre['fpol']       = int(fibre['DeltaThetaC']**2 * tx['Rs']/4/rx["NSymbFrame"]/rx["SymbScale"])
    
    print('======= set-Nsymbols sum up =======')
    print('------- general:')
    print('symb scale           = {}'.format(rx["SymbScale"]))
    print('Npol = NsymbFrame    = {}'.format(rx["NSymbFrame"]))
    print('Nphi = NSymbBatch    = {}'.format(rx["NSymbBatch"]))
    print('NSymb_Added_Net      = {}'.format(Nsymb_added_net))
    print('Npol/Nphi            = {}'.format(rx["NSymbFrame"]/rx["NSymbBatch"]))
    print('NsampTaps            = {}'.format(tx["NsampTaps"]))
    
    # spaces such as the printed lines on the shell are aligned
    if rx['mode'].lower() != "blind":
        print('\n------- if header:')
        print(f"NSymbs_pilots    = {NSymbs_pilots}")

        # print('Nsymb_data_Batch         = {}'.format(rx["NSymb_data_Batch"]))
        # print('NSymb_pilots_tot_Batch   = {}'.format(rx['NSymb_pilots_tot_Batch']))
        # print('NSymb_overhead_percent   = {}'.format(rx['NSymb_overhead_percent']))
        # print('Effective baud rate      = {}'.format(rx['Rs_eff']))

    print('\n------- physics:')
    print("fpol                 = {}".format(fibre['fpol']))
    print('dnu                  = {}'.format(tx['dnu']))
    print('===================================')



    tx              = misc.sort_dict_by_keys(tx)
    fibre           = misc.sort_dict_by_keys(fibre)
    rx              = misc.sort_dict_by_keys(rx)
    return tx,fibre,rx



#%%

def shaping_filter(tx):

    if "shaping_filter" not in tx:
        tx['shaping_filter'] = "dirac"

    if tx['shaping_filter'].lower() == 'dirac':
        h               = np.zeros(tx['NsampTaps'])
        i               = round(tx['NsampTaps']/2)
        h[i]            = 1

    elif tx['shaping_filter'].lower() == 'random':
        np.random.seed(0)
        h       = np.random.rand(tx['NsampTaps'])/3
        i       = round(tx['NsampTaps']/2)
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

    tx['hmatrix']       = np.zeros((4,tx['NsampTaps']))
    tx['hmatrix'][0]    = h
    tx['hmatrix'][3]    = h

    # gen.plot_fir(tx,tx['hmatrix'])

    tx      = misc.sort_dict_by_keys(tx)

    return tx

