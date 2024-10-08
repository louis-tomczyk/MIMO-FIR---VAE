# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 2.0.1
#   Date            : 2024-07-24
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2023-03-04) - creation
#   1.0.1 (2024-04-21) - set_thetas: if 'theta_in' not in fibre["ThetasLaw"]
#   1.0.2 (2024-04-24) - set_thetas: as 04-21, propagation: awgn
#   1.0.3 (2024-05-27) - simulate dispersion: C/PM-D ---> tauC/PM-D
#   1.0.4 (2024-06-27) - propagation: varargin to check synchro of pilots
#   1.0.5 (2024-07-10) - naming normalisation (*frame*-> *Frame*).
#                        along with main (1.4.3)
# ---------------------
#   2.0.0 (2024-07-12) - LIBRARY NAME CHANGED: LIB_GENERAL -> LIB_PLOT
#   2.0.1 (2024-07-24) - server management
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
#   CODE
#   [C1] Author          : Vincent Lauinger
#        Contact         : vincent.lauinger@kit.edu
#        Laboratory/team : Communications Engineering Lab
#        Institution     : Karlsruhe Institute of Technology (KIT)
#        Date            : 2022-06-15
#        Program Title   : 
#        Code Version    : 
#        Web Address     : https://github.com/kit-cel/vae-equalizer
# ---------------------------------------------
# %%



#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import torch
import lib_misc as misc
from lib_misc import KEYS as keys
import lib_rxhw as rxhw
import lib_prop as prop
import lib_plot as plot

pi = np.pi
from lib_matlab import clc


#%% ===========================================================================
# --- CONTENTS
# =============================================================================
# - gen_random_theta            --- called in : prop.set_thetas
# - propagation                 --- called in : processing.processing
# - set_thetas                  --- called in : processing.init_processing
# - simulate_dispersion         --- called in : prop.propagation
# =============================================================================


#%% # linear channel
def propagation(tx,fibre,rx,*varargin):

    rx      = simulate_dispersion(tx,fibre,rx)
    rx      = rxhw.load_ase(tx,rx)
    
    # ---------------------------------------------------------------- to check
    if len(varargin) > 0 and (varargin is not None) and (not rx['server']):
        # cf. data_shaping, convolution with filter, mode valid
        # 
        # y  = conv(x,h,'valid)
        # len(x) = N,   len(h) = M,     len(y) = N-M+1
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
    
            plt.suptitle("data shaping {} - {}-{}".format(rx['Frame'],kstart,kend))
            plt.show()
    

    # if rx['Frame'] >= rx['FrameChannel']:
    #     plot.constellations(rx['sig_real'], "prop  f-{}".format(rx['Frame']-rx['FrameChannel']),tx)
    # ---------------------------------------------------------------- to check
    
    rx      = misc.sort_dict_by_keys(rx)
    
    return tx,fibre,rx

#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================

#%%
def gen_random_theta(fibre):

    ThetasLaw           = fibre['ThetasLaw']
    law                 = ThetasLaw["law"]
    numelvalid          = ThetasLaw["numelvalid"]+1
    output              = np.zeros(numelvalid+1)
    output[0]           = -ThetasLaw["theta_in"]
    
    if ThetasLaw["kind"] == "Rwalk":

        if law == "uni":
            low         = ThetasLaw['low']
            high        = ThetasLaw['high']
            output      = np.cumsum(np.random.uniform(low,high,numelvalid))


        elif law == "gauss":
            mean        = 0
            std         = ThetasLaw['theta_std']
            output      = np.cumsum(np.random.normal(mean,std,numelvalid))


        elif law == "expo":
            mean        = ThetasLaw['mean']
            output      = np.cumsum(np.random.exponential(mean,numelvalid))


        elif law == "tri":
            low         = ThetasLaw['low']
            high        = ThetasLaw['high']
            mode        = ThetasLaw['mode']
            output      = np.cumsum(np.random.triangular(low,mode,high))


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








#%%
def set_thetas(tx,fibre,rx):

    fibre["ThetasLaw"]["numel"]         = rx["NFrames"]
    numel                               = rx["NFrames"]

    if "FrameChannel" not in rx:
        fibre["ThetasLaw"]["numeltrain"]= int(np.floor(rx["NFrames"]/2))
    else:
        fibre["ThetasLaw"]["numeltrain"]=  rx["FrameChannel"]

    fibre["ThetasLaw"]["numelvalid"]    = rx["NFrames"]-fibre["ThetasLaw"]["numeltrain"] 

    numeltrain                          = int(fibre["ThetasLaw"]["numeltrain"])
    numelvalid                          = int(fibre["ThetasLaw"]["numelvalid"])

    # if nothing specified, by default put random varying theta
    if "kind" not in fibre["ThetasLaw"]:
        fibre["ThetasLaw"]["kind"]      = "Rwalk"
        fibre["ThetasLaw"]["law"]       = "gaussian"

    # initialisation of the angles
    fibre["thetas"]                     = np.zeros((rx['NFrames'],tx["Npolars"]))

    # 1: first column is the input angles that are fixed to the 1st training angle phase
    # 2: second colum is filled with the opposite angles to have only PMD and CD
    #    up to half of the number of frames [can be changed later]
    # 3: generation of the random angles following the random parameters
    #    and filling the 2nd training angle phase

    if fibre['channel'].lower() != "awgn":
        if 'theta_in' not in fibre["ThetasLaw"]:
            fibre["ThetasLaw"]['theta_in']  = fibre["ThetasLaw"]['Start']
    
        fibre["thetas"][:,0]                = fibre["ThetasLaw"]['theta_in']           #1
        fibre["thetas"][:numeltrain,1]      = -fibre["ThetasLaw"]['theta_in']          #2
    
        if "thetadiffs" not in fibre:
            if fibre["ThetasLaw"]["kind"] == "Rwalk":
                fibre["thetas"][numel-numelvalid:,1]    = prop.gen_random_theta(fibre)   #3
                
            if fibre["ThetasLaw"]["law"] == "lin":
                ThetaStart                              = fibre["ThetasLaw"]["Start"]
                ThetaEnd                                = fibre["ThetasLaw"]["End"]
                fibre["ThetasLaw"]["Sth"]               = (ThetaEnd-ThetaStart)/numelvalid
                fibre["thetas"][numel-numelvalid:,1]    = np.linspace(ThetaStart+fibre["ThetasLaw"]["Sth"],ThetaEnd-ThetaStart, numelvalid)
    
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
def simulate_dispersion(tx,fibre,rx):

    if fibre['channel'].lower() != "awgn":
        theta               = np.array(fibre["thetas"][rx["Frame"]])                # [rad]

        tx_sig_cplx         = np.zeros((tx['Npolars'],tx['NsampFrame']),dtype = np.complex64)
        tx_sig_cplx[0]      = np.array(tx['sig_real'][0]+1j*tx['sig_real'][1])
        tx_sig_cplx[1]      = np.array(tx['sig_real'][2]+1j*tx['sig_real'][3])

        tx_sig_fft          = np.fft.fft(tx_sig_cplx, axis=1).astype(np.complex64)
        omega               = np.fft.fftfreq(tx["NsampFrame"], 1/tx["fs"]).astype(np.float32)*2*pi   # [rad/s]

        exp_cd              = np.exp(1j*omega**2*fibre["tauCD"]**2/2).astype(np.complex64)
        exp_pmd             = np.exp(1j*omega*fibre["tauPMD"]/2).astype(np.complex64)
        exp_phiIQ           = np.exp(-1j*fibre["phiIQ"]).astype(np.complex64)


        if len(theta) == 1:
            theta1          = theta[0]
            theta2          = -theta1
        else:
            theta1          = theta[0]
            theta2          = theta[1]

        R1                  = np.asarray([[np.cos(theta1)  , np.sin(theta1)],
                                          [-np.sin(theta1) , np.cos(theta1)]])

        R2                  = np.asarray([[np.cos(theta2)  , np.sin(theta2)],
                                          [-np.sin(theta2) , np.cos(theta2)]])

        Diag_pmd            = np.asarray([[exp_pmd*exp_phiIQ[0], 0],
                                  [0, 1/exp_pmd*exp_phiIQ[1]]], 
                                 dtype=object)

        h_gnd               = R2 @ Diag_pmd @ R1

        RX_fft              = np.zeros((2,tx["NsampFrame"]), dtype = np.complex64)
        RX_fft[0,:]         = (h_gnd[0,0]*tx_sig_fft[0,:] + h_gnd[0,1]*tx_sig_fft[1,:])*exp_cd
        RX_fft[1,:]         = (h_gnd[1,0]*tx_sig_fft[0,:] + h_gnd[1,1]*tx_sig_fft[1,:])*exp_cd 

        rx_sig_cplx         = np.fft.ifft(RX_fft, axis=1).astype(np.complex64)

        rx['sig_real'][0]   = torch.tensor(np.real(rx_sig_cplx[0]), dtype = torch.float32)
        rx['sig_real'][1]   = torch.tensor(np.imag(rx_sig_cplx[0]), dtype = torch.float32)
        rx['sig_real'][2]   = torch.tensor(np.real(rx_sig_cplx[1]), dtype = torch.float32)
        rx['sig_real'][3]   = torch.tensor(np.imag(rx_sig_cplx[1]), dtype = torch.float32)

        if rx['save_channel_gnd']:        
            rx["h_gnd"].append(h_gnd)

    else:

        rx['sig_real'][0]   = torch.tensor(tx['sig_real'][0], dtype = torch.float32)
        rx['sig_real'][1]   = torch.tensor(tx['sig_real'][1], dtype = torch.float32)
        rx['sig_real'][2]   = torch.tensor(tx['sig_real'][2], dtype = torch.float32)
        rx['sig_real'][3]   = torch.tensor(tx['sig_real'][3], dtype = torch.float32)

    rx          = misc.sort_dict_by_keys(rx)

    return rx

