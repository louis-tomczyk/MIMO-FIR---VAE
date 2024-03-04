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
import lib_rxhw as rxhw
import lib_prop as prop


pi = np.pi

#%%
# ============================================================================================ #
# computation of the PMD matrix, of the CD, an the Npolarsarisation rotation and phase IQ
# used later in generate_data_shaping
# simulate residual CD, PMD, Npolars. rot and IQ-shift in f-domain
# ============================================================================================ #
def simulate_dispersion(tx,fibre,rx): # in PROPAGATION

    theta       = np.array(fibre["thetas"][rx["Frame"]])                # [rad]
    tx_sig_fft  = np.fft.fft(tx["sig_cplx"], axis=1)        
    omega       = np.fft.fftfreq(tx["Nsamp_rx_tmp"], 1/tx["fs"])*2*pi   # [rad/s]
    
    exp_cd      = np.exp(1j*omega**2*fibre["TauCD"]/2)
    exp_pmd     = np.exp(1j*omega*fibre["TauPMD"]/2)
    exp_phiIQ   = np.exp(-1j*fibre["phiIQ"])

    
    if len(theta) == 1:
        theta1  = theta[0]
        theta2  = -theta1
    else:
        theta1  = theta[0]
        theta2  = theta[1]
        
    R1          = np.asarray([[np.cos(theta1)  , np.sin(theta1)],
                             [-np.sin(theta1)  , np.cos(theta1)]])

    R2          = np.asarray([[np.cos(theta2)  , np.sin(theta2)],
                             [-np.sin(theta2)  , np.cos(theta2)]])
    
    Diag_pmd    = np.asarray([[exp_pmd*exp_phiIQ[0], 0],
                              [0, 1/exp_pmd*exp_phiIQ[1]]], 
                             dtype=object)
    
    H_lin       = R2 @ Diag_pmd @ R1
    
    RX_fft      = np.zeros((2,tx["Nsamp_rx_tmp"]), dtype = np.complex128)
    RX_fft[0,:] = (H_lin[0,0]*tx_sig_fft[0,:] + H_lin[0,1]*tx_sig_fft[1,:])*exp_cd
    RX_fft[1,:] = (H_lin[1,0]*tx_sig_fft[0,:] + H_lin[1,1]*tx_sig_fft[1,:])*exp_cd 
    
    rx["sig_cplx"]  = np.complex128(np.fft.ifft(RX_fft, axis=1))
    rx["Hlin"]      = H_lin
    rx["H_lins"].append(rx["Hlin"])
    
    rx          = misc.sort_dict_by_keys(rx)
    
    return rx






#%%
# =============================================================================
# simulates the propagation in the channel with PMD, CD
# =============================================================================
def propagation(tx,fibre,rx): # before TRAINING PER MINIBATCH

    # linear channel
    rx_disp     = simulate_dispersion(tx,fibre,rx)
    rx          = rxhw.load_ase(tx,rx_disp)
    
    tx          = misc.sort_dict_by_keys(tx)
    fibre       = misc.sort_dict_by_keys(fibre)
    rx          = misc.sort_dict_by_keys(rx)
    
    return tx,fibre,rx


#%%
def set_thetas(tx,fibre,rx):
        
    fibre["ThetasLaw"]["numel"]         = rx["Nframes"]
    numel                               = rx["Nframes"]
    
    
    isFrameRndRotPresent                = "FrameRndRot" in rx
    if isFrameRndRotPresent == False:
        fibre["ThetasLaw"]["numeltrain"]= int(np.floor(rx["Nframes"]/2))
    else:
        fibre["ThetasLaw"]["numeltrain"]=  rx["FrameRndRot"]
    
    
    fibre["ThetasLaw"]["numelvalid"]    = rx["Nframes"]-fibre["ThetasLaw"]["numeltrain"] 
    
    numeltrain  = int(fibre["ThetasLaw"]["numeltrain"])
    numelvalid  = int(fibre["ThetasLaw"]["numelvalid"])

    # if nothing specified, by default put random varying theta    
    isKindPresent = "kind" in fibre["ThetasLaw"]
    if isKindPresent == False:
        fibre["ThetasLaw"]["kind"]      = "Rwalk"
        fibre["ThetasLaw"]["law"]       = "gaussian"
        
    # initialisation of the angles
    fibre["thetas"]                 = np.zeros((rx['Nframes'],tx["Npolars"]))

    # 1: first column is the input angles that are fixed to the 1st training angle phase
    # 2: second colum is filled with the opposite angles to have only PMD and CD
    #    up to half of the number of frames [can be changed later]
    # 3: generation of the random angles following the random parameters
    #    and filling the 2nd training angle phase
    
    fibre["thetas"][:,0]            = fibre["ThetasLaw"]['theta_in']           #1
    fibre["thetas"][:numeltrain,1]  = -fibre["ThetasLaw"]['theta_in']          #2
    
    
    isThetaDiffsPresent             = "thetadiffs" in fibre
    if isThetaDiffsPresent == False:
        if fibre["ThetasLaw"]["kind"] == "Rwalk":
            fibre["thetas"][numel-numelvalid:,1]  = prop.gen_random_theta(fibre)   #3
            
        if fibre["ThetasLaw"]["law"] == "lin":
            ThetaStart                          = fibre["ThetasLaw"]["Start"]
            ThetaEnd                            = fibre["ThetasLaw"]["End"]
            fibre["ThetasLaw"]["slope"]         =  (ThetaEnd-ThetaStart)/numelvalid
            fibre["thetas"][numel-numelvalid:,1]= np.linspace(ThetaStart,ThetaEnd, numelvalid)
            
        # obtention of angle shifts
        fibre["thetadiffs"]             = fibre["thetas"][numelvalid:,1]-fibre["ThetasLaw"]['theta_in']
    
        # conversion into tensors
        fibre["thetas"]                 = misc.my_tensor(fibre["thetas"])
        fibre["thetadiffs"]             = misc.my_tensor(fibre["thetadiffs"])
    else:
        fibre["thetas"][numelvalid:,1]  =  fibre["thetadiffs"]
    
    
    fibre   = misc.sort_dict_by_keys(fibre)
    
    return fibre





#%%
# ============================================================================================ #
# GEN_RANDOM_THETA
# ============================================================================================ #
def gen_random_theta(fibre):

    ThetasLaw   = fibre['ThetasLaw']
    law         = ThetasLaw["law"]
    numelvalid  = ThetasLaw["numelvalid"]+1
    output      = np.zeros(numelvalid+1)
    output[0]   = -ThetasLaw["theta_in"]
    
    if ThetasLaw["kind"] == "Rwalk":

        if law == "uni":
            low     = ThetasLaw['low']
            high    = ThetasLaw['high']
            output  = np.cumsum(np.random.uniform(low,high,numelvalid))
                                
            # for k in range(1,numelvalid+1):
            #     output[k] = output[k-1]+np.random.uniform(low,high,1)
                
        elif law == "gauss":
            mean   = ThetasLaw['theta_in']
            std    = ThetasLaw['theta_std']
            output = np.cumsum(np.random.normal(mean,std,numelvalid))
            
            # for k in range(1,numelvalid+1):
            #     output[k] = output[k-1]+np.random.normal(mean,std)
                
        elif law == "expo":
            mean   = ThetasLaw['mean']
            output = np.cumsum(np.random.exponential(mean,numelvalid))

            # for k in range(1,numelvalid):
            #     output[k] = output[k-1]+np.random.exponential(mean,numelvalid)
                
        elif law == "tri":
            low     = ThetasLaw['low']
            high    = ThetasLaw['high']
            mode    = ThetasLaw['mode']
            output = np.cumsum(np.random.triangular(low,mode,high))
            
            # for k in range(1,numelvalid+1):
            #     output[k] = output[k-1]+np.random.triangular(low,mode,high)
        
        output[0:-1]    = output[1:]
        output          = np.delete(output,-1) # last element is repeated
        
    else: # random variable

        for k in range(numelvalid):
            if law == "uni":
                low     = ThetasLaw['low']
                high    = ThetasLaw['high']
                theta2  = np.random.uniform(low,high)
            elif law == "gauss":
                mean    = ThetasLaw['mean']
                std     = ThetasLaw['std']
                theta2  = np.random.normal(mean,std)

        output   = -theta2
        
    return output
