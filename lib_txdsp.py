# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : Louis Tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.2.0
#   Date            : 2024-06-10
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute - place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source - include copyright -
#                               state changes - include license
#
# ----- CHANGELOG -----
#   1.0.0 (2024-03-04)  - creation
#   1.0.3 (2024-04-03)  - cleaning
#   1.0.4 (2024-05-27)  - get_constellation: inspiration from [C2]
#   1.1.0 (2024-06-04)  - get_constellation: adding qualifying adjective (pilots,)
#                       - set_Nsymbols: displaying net symbols added/removed
#                       - [NEW] pilot_generation
#                       - transmitter: adding pilots management
#   1.1.1 (2024-06-07)  - set_Nsymbols, cleaning (Ntaps, Nconv)
#   1.2.0 (2024-06-07)  - pilot_generation: scaling to data power
#                       - [NEW] pilot_insertion
#                       - transmitter: including pilot_insertion
#
# ----- MAIN IDEA -----
#   Library for Digital Signal Processing at the Transmitter side in (optical) telecommunications
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   Authors             : 
#   Title               :
#   Journal/Editor      : 
#   Volume - N°         : 
#   Date                : 
#   DOI/ISBN            : 
#   Pages               : 
#
#   Functions:
#   [C1] Author         : Vincent Lauinger
#       Contact         : vincent.lauinger@kit.edu
#       Affiliation     : Communications Engineering Lab, Karlsruhe Institute of Technology (KIT)
#       Date            : 2022-06-15
#       Program Title   : 
#       Code Version    : 
#       Type            : Source code
#       Web Address     : https://github.com/kit-cel/vae-equalizer
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
import sys
import lib_misc as misc
import lib_general as gen
import lib_txhw as txhw
import lib_matlab as mb
pi = np.pi

from lib_misc import KEYS as keys
from lib_matlab import clc



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



#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================

#%%
def data_generation(tx,rx,*varargin):

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
    data_I          = data[0::tx["Npolars"],:] # skip using '::<number of rows to skip>
    data_Q          = data[1::tx["Npolars"],:]

    # sps-upsampled signal by zero-insertion
    # louis, do not remove the type definition
    tx["sig_cplx_up"]                   = np.zeros((tx["Npolars"],tx["Nsamp_up"]),dtype=np.complex64)
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
def data_shaping(tx):

    # # The "valid" mode for convolution already shrinks the length
    # tx["NsampFrame"]  = tx["Nsamp_up"]-(tx['NsampTaps']-1) # = 10,377-13+1=10,365
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
def get_constellation(tx,rx,*varargin):
    
    ################################################################################ 
    ########################## CONSTELLATION CONSTRUCTION ##########################
    ################################################################################
    
    # parameters definition of each constellation
    constellations = {}
    params = {
        '4QAM'  :   (2, [-1, 1]),
        '16QAM' :   (4, [-3, -1, 1, 3]),
        '64QAM' :   (8, [-7, -5, -3, -1, 1, 3, 5, 7]),
    }
    
    if len(varargin) != 0:
        pilots_name = 'pilots_'+varargin[0][0]
        pilots_mod  = varargin[0][-2]
        
    
    if len(varargin) == 0:
        Nradii = {
            '4QAM'  :   (1, [4]),
            '16QAM' :   (3, [4, 8, 4]),
            '64QAM' :   (9, [4, 8, 4, 8, 8, 12, 8, 8, 4])
        }
    
    # creation of available constellations
    for modfor, (side, amplitudes) in params.items():
        points = []
        for i in range(side):
            for j in range(side):
                points.append(amplitudes[i] + 1j * amplitudes[j])
        constellations[modfor]  = np.array(points)


    # norm factor: sqrt(2*(M-1)/3)
    # 4qam: sqrt(2) --- 16 qam: sqrt(10) --- 64qam: sqrt(42) --- 256qam: sqrt(170)
    if len(varargin) == 0:
        tx["const_affixes"]     = constellations[tx["mod"]]
    else:
        tx["const_affixes_{}".format(pilots_name)]  = constellations[pilots_mod]
        
    if len(varargin) == 0:
        rings                   = np.unique(abs(tx["const_affixes"]))
        Prob_symb_tmp           = np.exp(-tx['nu']*rings**2)
        Prob_ring_tmp           = Prob_symb_tmp*Nradii[tx['mod']][1]
        Prob_symb               = Prob_symb_tmp/sum(Prob_ring_tmp)
        Prob_ring               = Prob_symb*Nradii[tx['mod']][1]
        
        Symb_Probs_tmp          = np.exp(-tx['nu']*abs(tx["const_affixes"]))
        Symb_Probs              = Symb_Probs_tmp/sum(Symb_Probs_tmp)

    if len(varargin) == 0:
        norm_factor             = np.sqrt(np.mean(np.abs(tx["const_affixes"])**2))
        constellation           = tx["const_affixes"]/norm_factor
    else:
        norm_factor_pilots      = np.sqrt(np.mean(np.abs(tx["const_affixes_{}".format(pilots_name)]**2)))
        constellation_pilots    = tx["const_affixes_{}".format(pilots_name)]/norm_factor_pilots
        
    # misc.plot_complex_constellation(constellation)
    
    ################################################################################
    ################################ PROBABILITY LAW ###############################
    ################################################################################

    # extract real amplitudes

    if  len(varargin) == 0:
        N_amps          = int(np.sqrt(len(constellation.real))) # number of ASK levels
        amps            = constellation.real[::N_amps]          # amplitude levels
        sc              = min(abs(amps))                        # scaling factor for having lowest level equal 1
        nu_sc           = tx['nu']/(sc**2)
    
        # probabilities of the amlitude levels
        P               = np.exp(-tx["nu"]*np.abs(amps/sc)**2)
        P               = P / sum(P)  
        P               = np.expand_dims(P, axis=0)
        T               = np.dot(P.T,P)
        P               = P.reshape(-1)
        
    else:
        N_amps_pilots   = int(np.sqrt(len(constellation_pilots.real)))
        amps_pilots     = constellation_pilots.real[::N_amps_pilots]

    # mb.imagesc(np.log10(T)) # plot the probability matrix
    ################################################################################
    #################################### SAVING ####################################
    ################################################################################

    if  len(varargin) == 0:
        pow_mean = np.sum(T.reshape(-1)* np.abs(constellation)**2)
    else:
        pow_mean_pilots = np.sum(np.abs(constellation_pilots)**2)

    if  len(varargin) == 0:
        tx["const_norm_factor"] = norm_factor
        tx["constellation"]     = np.expand_dims(constellation,axis = 1)
        tx["N_amps"]            = N_amps                    # number of positive amplitude
        tx["pow_mean"]          = pow_mean                  # mean power of the constellation
        tx["Symb_Probs"]        = Symb_Probs
        tx['Prob_ring']         = Prob_ring
        tx['nu_sc']             = nu_sc
    
        if rx['mimo'].lower() == "vae":
            tx["amps"]          = torch.tensor(amps)
            tx["prob_amps"]     = torch.tensor(P)       # probabilities of the amplitude levels
        else:
            tx["amps"]          = amps
            tx["prob_amps"]     = P
    else:
        tx["const_norm_factor_{}".format(pilots_name)]= norm_factor_pilots
        tx["constellation_{}".format(pilots_name)]    = np.expand_dims(constellation_pilots,axis = 1)
        tx["N_amps_{}".format(pilots_name)]           = N_amps_pilots
        tx["pow_mean_{}".format(pilots_name)]         = pow_mean_pilots
    
        if rx['mimo'].lower() == "vae":
            tx["amps_{}".format(pilots_name)]         = torch.tensor(amps_pilots)
        else:
            tx["amps_{}".format(pilots_name)]         = amps_pilots

        
    tx  = misc.sort_dict_by_keys(tx)
    rx  = misc.sort_dict_by_keys(rx)

    return tx,rx


#%%
# varagin[0] {pilots_cpr,pilot_synchro}, if _synchro => cazac sequences
# varagin[1] {same, different} for {same} or {different} pilots symbols for each polarisation

def pilot_generation(tx,rx,what_pilots):

    ################################################################################
    ################################# maintenance ##################################
    ################################################################################
    
    if rx['mode'].lower() != "blind":
    
        pilots_dsp          = 'pilots_'+what_pilots[0]
        pilots_selection    = what_pilots[1]
        pilots_batch_wise   = what_pilots[2]
        pilots_polar_wise   = what_pilots[3]
        pilots_percentage   = what_pilots[-1]/100
    
        if (pilots_batch_wise.lower() == 'fixed')\
            and '{}_flag_same_all_batches'.format(pilots_dsp) in tx:
            
                flag_do = 0
                
        elif (pilots_batch_wise.lower() == 'fixed')\
            and '{}_flag_same_all_batches'.format(pilots_dsp) not in tx:
                tx['{}_flag_same_all_batches'.format(pilots_dsp)] = 1
    
                flag_do = 1
                
        elif pilots_batch_wise.lower() != 'fixed':
            flag_do = 1
            
        ################################################################################
        ################################## generation ##################################
        ################################################################################
        if flag_do:
            tx['NSymb_{}_Batch'.format(pilots_dsp)] = int(round(pilots_percentage*tx["NSymbConv"]/rx['NBatchFrame'],0))
            tx["Nsamp_{}_Batch".format(pilots_dsp)] = tx["Nsps"]*tx["NSymb_{}_Batch".format(pilots_dsp)]
    
            if pilots_selection.lower() == "rand":
                if pilots_polar_wise.lower() == "same":
                    pilots      = np.random.default_rng().choice(\
                                    tx["amps_{}".format(pilots_dsp)],
                                    (2,tx["NSymb_{}_Batch".format(pilots_dsp)]))
                    pilots      = mb.repmat(pilots, (2,1))
                else:
                    pilots      = np.random.default_rng().choice(\
                                    tx["amps_{}".format(pilots_dsp)],
                                    (2*tx['Npolars'],tx["NSymb_{}_Batch".format(pilots_dsp)]))
            else:
                print('not implemented yet')
                exit
    
            pilots_I    = pilots[0::tx["Npolars"],:]
            pilots_Q    = pilots[1::tx["Npolars"],:]
            
            pow_I       = np.mean(np.abs(pilots_I)**2)
            pow_Q       = np.mean(np.abs(pilots_Q)**2)
            tx['{}_power'.format(pilots_dsp)] = pow_I+pow_Q
    
            power_scale = np.sqrt(tx['sig_power']/tx['{}_power'.format(pilots_dsp)])
            pilots_I    *= power_scale
            pilots_Q    *= power_scale
            
            tx["{}_cplx_up".format(pilots_dsp)]     = \
            np.zeros((tx["Npolars"],tx["Nsamp_{}_Batch".format(pilots_dsp)]),dtype=np.complex64)
                
            tx["{}_cplx_up".format(pilots_dsp)][:,::tx["Nsps"]]   = pilots_I + 1j*pilots_Q
            
            # gen.plot_constellations(tx['{}_cplx_up'.format(pilots_dsp)],polar='both',sps=1)    
            tx  = misc.sort_dict_by_keys(tx)
            
    return tx


#%%
# [synchro, cazac, pilots]
def pilot_insertion(tx,rx,what_pilots):    
    
    if rx['mode'].lower() != "blind":
        if what_pilots[0].lower() == "synchro":
            print('not implemented yet')
            exit
        elif what_pilots[0].lower() == "cazac":
            print('not implemented yet')
            exit
        else:
            for k in range(rx['NBatchFrame']):
                index_start = 0+k*rx['NsampBatch']
                index_end   = index_start + tx['Nsamp_{}_Batch'.format('pilots_'+what_pilots[0])]
                tx['sig_cplx_up'][:,index_start:index_end] = tx['pilots_{}_cplx_up'.format(what_pilots[0])]
        
        
    return tx,rx

#%%
def set_Nsymbols(tx,fibre,rx):

    # number of symbols having the same polarisation state
    if fibre['fpol'] != 0:
        rx["NSymbFrame"]    = int(fibre['DeltaThetaC']**2/4*tx['Rs']/fibre['fpol'])
    else:
        rx["NSymbFrame"]    = 10000
    
    # number of symbols having more or less the same phase
    if tx['dnu'] != 0:
        rx["NSymbBatch"]    = int(tx['DeltaPhiC']**2/4*tx['Rs']/tx['dnu'])
    else:
        rx['NSymbBatch']    = 100
    
    tx['NSymbFrame']        = rx["NSymbFrame"]
    tx["NSymbConv"]             = rx["NSymbFrame"]+tx['NSymbTaps']+1
    tx["Nsamp_up"]          = tx["Nsps"]*(tx["NSymbConv"]-1)+1
    tx['NsampTaps']         = tx["Nsps"]*tx["NSymbTaps"]-1      # length of FIR filter
    tx["NsampFrame"]        = tx["Nsamp_up"]-(tx['NsampTaps']-1)

    ################################################################################
    ########## adjustment of the number of symbols for phase/pol dynamics ##########
    ################################################################################
    
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
        
    if rx['mode'].lower() != "blind":
        Nsymbs_pilots = 0
        for k in range(len(tx['pilots_info'])):

            tmp                                 = tx['pilots_info'][k]
            name                                = 'pilots_' + tmp[0]
            
            rx['NSymb_{}_Batch'.format(name)]   = int(round(tmp[-1]/100*rx['NSymbBatch']))
            tx['NSymb_{}_Batch'.format(name)]   = rx['NSymb_{}_Batch'.format(name)]
            Nsymbs_pilots                       += tx['NSymb_{}_Batch'.format(name)]
            
        rx['NSymb_data_Batch']          = rx['NSymbBatch']-Nsymbs_pilots
        rx['NSymb_pilots_tot_Batch']    = Nsymbs_pilots
        rx['NSymb_overhead_percent']    = round(rx['NSymb_pilots_tot_Batch']/rx['NSymb_data_Batch']*100,2)
        rx['Rs_eff']                    = round(rx['NSymb_data_Batch']/rx['NSymbBatch']*tx['Rs']*1e-9,2)
            


    ################################################################################
    ############################# updating the numbers #############################
    ################################################################################
    
    tx['NSymbFrame']        = rx["NSymbFrame"]
    tx["NSymbConv"]         = rx["NSymbFrame"]+tx['NSymbTaps']+1
    tx["Nsamp_up"]          = tx["Nsps"]*(tx["NSymbConv"]-1)+1
    tx['NsampTaps']         = tx["Nsps"]*tx["NSymbTaps"]-1      # length of FIR filter
    tx["NsampFrame"]        = tx["Nsamp_up"]-(tx['NsampTaps']-1)


    ################################################################################
    ############################ miscellaneous #####################################
    ################################################################################
    
    tx['NSymbTot']          = tx['NSymbFrame']*rx['Nframes']
    tx['NsampTot']          = tx['NsampFrame']*rx['Nframes']
    rx["NsampBatch"]        = rx['NSymbBatch']*tx['Nsps']    
    
    rx["NSymbCut"]          = 10  # number of symbols cut off to prevent edge effects of convolution
    rx["NSymbCut_tot"]      = 2*rx['NSymbCut']+1
    
    rx['NSymbEq']           = rx["NSymbFrame"]    
    if rx['mimo'].lower() != "vae":        
        rx['NSymbEq']       -= rx['NSymbCut_tot']-1

    ################################################################################
    ############################## displaying results ##############################
    ################################################################################
    
    if flag != 0:
        tx['dnu']           = int(tx['DeltaPhiC']**2 * tx['Rs']/4/rx["NSymbBatch"]/rx["SymbScale"])
        fibre['fpol']       = int(fibre['DeltaThetaC']**2 * tx['Rs']/4/rx["NSymbFrame"]/rx["SymbScale"])
        
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

            print('Nsymb_data_Batch         = {}'.format(rx["NSymb_data_Batch"]))
            print('NSymb_pilots_tot_Batch   = {}'.format(rx['NSymb_pilots_tot_Batch']))
            print('NSymb_overhead_percent   = {}'.format(rx['NSymb_overhead_percent']))
            print('Effective baud rate      = {}'.format(rx['Rs_eff']))

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

#%%
def transmitter(tx,rx):

    tx          = shaping_filter(tx)
    tx          = data_generation(tx,rx)
    
    for k in range(len(tx['pilots_info'])):
        tx      = pilot_generation(tx, rx, tx['pilots_info'][k])
        tx,rx   = pilot_insertion(tx,rx, tx['pilots_info'][k])
    gen.plot_constellations(tx['sig_cplx_up'])

    tx          = data_shaping(tx)
    
        
    if rx["Frame"] >= rx["FrameChannel"]:

        tx      = txhw.load_ase(tx,rx)
        tx      = txhw.load_phase_noise(tx,rx)

        
    tx          = misc.sort_dict_by_keys(tx)
    return tx






