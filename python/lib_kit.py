# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.1.6
#   Date            : 2024-11-08
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute -
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source -
#                               include copyright - state changes -
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2024-03-04) - creation
#   1.1.0 (2024-04-01) - cleaning
#   1.1.1 (2024-05-22) - CMA, use numpy instead of torch, saves memory & time
#   1.1.2 (2024-06-06) - [REMOVED] init_dict, moved to misc
#   1.1.3 (2024-06-21) - train_self -> train_vae
#   1.1.4 (2024-07-05) - cma, twoXtwoFIR: changing sig_eq_real -> sig_mimo_real
#                        along with rxdsp (1.6.2), processing (1.3.2)
#   1.1.5 (2024-07-10) - naming normalisation (*frame*-> *Frame*).
#                        along with main (1.4.3)
#   1.1.6 (2024-11-08) - Inf numpy
#
# ----- MAIN IDEA -----
#   Library for CMA equalizer in (optical) telecommunications
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   [A1] Authors        : Avi Caciularu
#       Title           : Blind Channel Equalization Using Variational
#                           Autoencoders
#       Journal/Editor  : ICC
#       Volume - N°     : 
#       Date            : May 2018
#       DOI/ISBN        : 10.1109/ICCW.2018.8403666
#       Pages           : 
#
#   [A2] Authors        : Junho Cho
#       Title           : Probabilistic Constellation Shaping for Optical Fiber
#                           Communications
#       Journal/Editor  : JLT
#       Volume - N°     : 37-6
#       Date            : March 2019
#       DOI/ISBN        : 10.1109/JLT.2019.2898855
#       Pages           : 
#
#   CODE:
#   [C1] Author          : Vincent Lauinger
#        Contact         : vincent.lauinger@kit.edu
#        Laboratory/team : Communications Engineering Lab
#        Institution     : Karlsruhe Institute of Technology (KIT)
#        Date            : 2022-06-15
#        Program Title   : 
#        Code Version    : 
#        Web Address     : https://github.com/kit-cel/vae-equalizer
# ---------------------------------------------


#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import lib_misc as misc
import lib_rxdsp as rxdsp
import lib_plot as plot

from lib_matlab import clc
import time

pi  = np.pi
Inf = np.inf

#%% ===========================================================================
# --- CONTENTS
# =============================================================================
# - CMA
# - CMAbatch
# - CMAflex
# - compute_vae_loss
# - CPE
# - dec_on_bound
# - find_shift
# - find_shift_symb_full
# - SER_constell_shaping
# - SER_estimation
# - SER_IQflip
# - soft_dec
# - train_vae
# - twoXtwoFIR
# =============================================================================


#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================


#%% [C3]
def CMA(tx,rx): # Constant Modulus Algorithm

    N       = rx["sig_real"].shape[-1]
    mh      = tx['NsampTaps']//2

    # maintenance
    if 'CMA' not in rx:
        rx['CMA']           = dict()
        rx['CMA']['losses'] = dict()

    if str(rx['Frame']) not in rx['CMA']['losses']:
        rx['CMA']['losses'][str(rx['Frame'])] = dict()

    if 'R' not in rx['CMA']:
        rx['CMA']['R']      = 1

    if type(rx["sig_real"]) != torch.Tensor:
        rx["sig_real"] = misc.my_tensor(rx["sig_real"], dtype = torch.float32)

    out     = np.zeros((2,2,N//tx['Nsps']))
    loss    = np.zeros((N//tx['Nsps'], 2))

    # zero-padding
    pad                 = np.zeros((2,2,mh))
    rx_sig_real         = np.zeros((2,2,len(rx['sig_real'].T)))

    rx_sig_real[0,0]    = rx['sig_real'][0]
    rx_sig_real[0,1]    = rx['sig_real'][1]
    rx_sig_real[1,0]    = rx['sig_real'][2]
    rx_sig_real[1,1]    = rx['sig_real'][3]

    y                   = np.concatenate((pad, rx_sig_real, pad), -1)
    y                   = y/np.mean(y[:,0,:]**2 + y[:,1,:]**2 )


    # training
    for i in np.arange(mh,N+mh,tx['Nsps']):
        ind = np.arange(-mh+i,i+mh+1)
        k   = i//tx['Nsps'] - mh

        # 2x2 butterfly FIR
        # Estimate Symbol 

        h11i    = rx['h_est'][0,0,0,:]
        h11q    = rx['h_est'][0,0,1,:]

        h22i    = rx['h_est'][1,1,0,:]
        h22q    = rx['h_est'][1,1,1,:]

        h12i    = rx['h_est'][0,1,0,:]
        h12q    = rx['h_est'][0,1,1,:]

        h21i    = rx['h_est'][1,0,0,:]
        h21q    = rx['h_est'][1,0,1,:]

        yhi     = y[0,0,ind]
        yhq     = y[0,1,ind]
        yvi     = y[1,0,ind]
        yvq     = y[1,1,ind]

        out[0,0,k] = ( # HI out
            np.matmul(yhi,h11i)  + np.matmul(yvi,h12i) -
            np.matmul(yhq,h11q)  - np.matmul(yvq,h12q)
        )
        
        # np.matmul(yhi,h11i) == NspT multiplications + NspT additions = 2*NspT
        #
        # np.matmul(yhi,h11i)  + np.matmul(yvi,h12i) -
        # np.matmul(yhq,h11q)  - np.matmul(yvq,h12q)
        #            == Npol*Nch matmul + Npol*Nch additions
        #            == Npol*Nch*(2*NspT)+ Npol*Nch = Npol*Nch*(2*NSpT+1)
        #
        # out[0,0,k] =  1 affectation
        #
        # out[0,0,k] = ( # HI out
        #     np.matmul(yhi,h11i)  + np.matmul(yvi,h12i) -
        #     np.matmul(yhq,h11q)  - np.matmul(yvq,h12q)
        # ) ======> Npol*Nch*(2*NSpT+1)+1

        out[1,0,k] = ( # VI out
            np.matmul(yhi,h21i)  + np.matmul(yvi,h22i) -
            np.matmul(yhq,h21q)  - np.matmul(yvq,h22q)
        )

        out[0,1,k] = ( # HQ out
          np.matmul(yhq,h11i)    + np.matmul(yvq,h12i) +
          np.matmul(yhi,h11q)    + np.matmul(yvi,h12q)
        )

        out[1,1,k] = ( # VQ out
            np.matmul(yhq,h21i)  + np.matmul(yvq,h22i) +
            np.matmul(yhi,h21q)  + np.matmul(yvi,h22q)
        )
        
        
# =============================================================================
#         #  out[...,,k] =======> Npol*Nch*[Npol*Nch*(2*NSpT+1)+1]
# =============================================================================

        # Calculate error
        loss[k,0] = rx['CMA']['R']**2 - out[0,0,k]**2 - out[0,1,k]**2
        loss[k,1] = rx['CMA']['R']**2 - out[1,0,k]**2 - out[1,1,k]**2
        
        # out[0,0,k]**2 == 1 product
        # rx['CMA']['R'] - out[0,0,k]**2 - out[0,1,k]**2 == 1+Npol additions
        # loss[k,0] = ... == 1 affection
        #
        # loss[k,0] = rx['CMA']['R'] - out[0,0,k]**2 - out[0,1,k]**2
        # ======> Npol+Npol+1+1 = 2*(Npol+1)
        
# =============================================================================
#         # loss ==============> 2*(Npol+1)
# =============================================================================

        rx['h_est'][0,0,0,:] += 2*rx['lr']*loss[k,0]* (out[0,0,k]*y[0,0,ind] + out[0,1,k]*y[0,1,ind])   # h11i
        rx['h_est'][0,0,1,:] += 2*rx['lr']*loss[k,0]* (out[0,1,k]*y[0,0,ind] - out[0,0,k]*y[0,1,ind])   # h11q 
        
        rx['h_est'][0,1,0,:] += 2*rx['lr']*loss[k,0]* (out[0,0,k]*y[1,0,ind] + out[0,1,k]*y[1,1,ind])   # h12i
        rx['h_est'][0,1,1,:] += 2*rx['lr']*loss[k,0]* (out[0,1,k]*y[1,0,ind] - out[0,0,k]*y[1,1,ind])   # h12q

        rx['h_est'][1,0,0,:] += 2*rx['lr']*loss[k,1]* (out[1,0,k]*y[0,0,ind] + out[1,1,k]*y[0,1,ind])   # h21i
        rx['h_est'][1,0,1,:] += 2*rx['lr']*loss[k,1]* (out[1,1,k]*y[0,0,ind] - out[1,0,k]*y[0,1,ind])   # h21q
        
        rx['h_est'][1,1,0,:] += 2*rx['lr']*loss[k,1]* (out[1,0,k]*y[1,0,ind] + out[1,1,k]*y[1,1,ind])   # h22i
        rx['h_est'][1,1,1,:] += 2*rx['lr']*loss[k,1]* (out[1,1,k]*y[1,0,ind] - out[1,0,k]*y[1,1,ind])   # h22q

        # out[0,0,k]*y[0,0,ind] + out[0,1,k]*y[0,1,ind] === Npol*NspT operations
        # += 2*rx['lr']*loss[k,0]*(out ...) === 3 mutliplications +1 addition
        #                                       + 1 affectation
        #       ====> (Npol*NspT+4)*Npol*Nch (Nch = 2 === I/Q)
        
# =============================================================================
#         #  h_est ===============>  (Npol*NspT+4)*Npol*Nch 
# =============================================================================

    rx['CMA']['losses'][str(rx['Frame'])] = loss


    rx['sig_mimo_real']       = np.zeros((4,out.shape[-1]))
    rx['sig_mimo_real'][0]    = out[0,0]
    rx['sig_mimo_real'][1]    = out[0,1]
    rx['sig_mimo_real'][2]    = out[1,0]
    rx['sig_mimo_real'][3]    = out[1,1]
    
    
    return rx, loss


# Npol*Nch*[Npol*Nch*(2*NSpT+1)+1] + 2*(Npol+1) +  (Npol*NspT+4)*Npol*Nch =
# 2*2*[2*2*(2*NSpT+1)+1]+ 2*(2+1) + (2*NspT+4)*2*2 =
# 4 * [4*(2*NspT+1)+1] + 6 + (NspT+2)*8
# 32*NspT+16+6+8*NspT+16
#  40*NspT + 42
# O(NspT)



#%%
def CMAbatch(Rx, R, h, lr, NSymbBatch, sps, eval):
    # device = Rx.device
    M = h.shape[-1]
    N = Rx.shape[-1]
    mh = M//2       

    # zero-padding
    pad = misc.my_zeros_tensor(2,2,mh)
    y   = torch.cat((pad, Rx, pad), -1)
    y   = y/torch.mean(y[:,0,:]**2 + y[:,1,:]**2 )

    out = misc.my_zeros_tensor(2,2,N//sps)
    e   = misc.my_zeros_tensor(N//sps, 2)
    buf = misc.my_zeros_tensor(2,2,2,N//sps)

    for i in torch.arange(mh,N+mh,sps):     # downsampling included
        ind = torch.arange(-mh+i,i+mh+1)
        k = i//sps - mh

        # 2x2 butterfly FIR
        out[0,0,k] = torch.matmul(y[0,0,ind],h[0,0,0,:]) - torch.matmul(y[0,1,ind],h[0,0,1,:])\
                    + torch.matmul(y[1,0,ind],h[0,1,0,:]) - torch.matmul(y[1,1,ind],h[0,1,1,:])

        out[1,0,k] = torch.matmul(y[0,0,ind],h[1,0,0,:]) - torch.matmul(y[0,1,ind],h[1,0,1,:])\
                    + torch.matmul(y[1,0,ind],h[1,1,0,:]) - torch.matmul(y[1,1,ind],h[1,1,1,:]) 
        
        out[0,1,k] = torch.matmul(y[0,0,ind],h[0,0,1,:]) + torch.matmul(y[0,1,ind],h[0,0,0,:])\
                    + torch.matmul(y[1,0,ind],h[0,1,1,:]) + torch.matmul(y[1,1,ind],h[0,1,0,:])

        out[1,1,k] = torch.matmul(y[0,0,ind],h[1,0,1,:]) + torch.matmul(y[0,1,ind],h[1,0,0,:])\
                    + torch.matmul(y[1,0,ind],h[1,1,1,:]) + torch.matmul(y[1,1,ind],h[1,1,0,:]) 

        e[k,0] = R - out[0,0,k]**2 - out[0,1,k]**2     # Calculate error
        e[k,1] = R - out[1,0,k]**2 - out[1,1,k]**2

        if eval == True:
            
            buf[0,0,0,k,:] = out[0,0,k]*y[0,0,ind] + out[0,1,k]*y[0,1,ind]          # buffering the filter update increments 
            buf[0,0,1,k,:] = out[0,1,k]*y[0,0,ind] - out[0,0,k]*y[0,1,ind]
            buf[0,1,0,k,:] = out[0,0,k]*y[1,0,ind] + out[0,1,k]*y[1,1,ind]
            buf[0,1,1,k,:] = out[0,1,k]*y[1,0,ind] - out[0,0,k]*y[1,1,ind]

            buf[1,0,0,k,:] = out[1,0,k]*y[0,0,ind] + out[1,1,k]*y[0,1,ind] 
            buf[1,0,1,k,:] = out[1,1,k]*y[0,0,ind] - out[1,0,k]*y[0,1,ind]
            buf[1,1,0,k,:] = out[1,0,k]*y[1,0,ind] + out[1,1,k]*y[1,1,ind]
            buf[1,1,1,k,:] = out[1,1,k]*y[1,0,ind] - out[1,0,k]*y[1,1,ind]      

            if (k%NSymbBatch == 0 and k!=0):  # batch-wise updating
                h[0,0,0,:] += 2*lr*torch.sum(torch.mul(buf[0,0,0,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,0]),dim=1)       # Update filters
                h[0,0,1,:] += 2*lr*torch.sum(torch.mul(buf[0,0,1,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,0]),dim=1) 
                h[0,1,0,:] += 2*lr*torch.sum(torch.mul(buf[0,1,0,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,0]),dim=1) 
                h[0,1,1,:] += 2*lr*torch.sum(torch.mul(buf[0,1,1,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,0]),dim=1) 

                h[1,0,0,:] += 2*lr*torch.sum(torch.mul(buf[1,0,0,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,1]),dim=1) 
                h[1,0,1,:] += 2*lr*torch.sum(torch.mul(buf[1,0,1,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,1]),dim=1) 
                h[1,1,0,:] += 2*lr*torch.sum(torch.mul(buf[1,1,0,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,1]),dim=1) 
                h[1,1,1,:] += 2*lr*torch.sum(torch.mul(buf[1,1,1,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,1]),dim=1) 
    return out, h, e

#%%
def CMAflex(Rx, R, h, lr, NSymbBatch, symb_step, sps, eval):
    device  = Rx.device
    M       = h.shape[-1]
    N       = Rx.shape[-1]
    mh      = M//2       

    # zero-padding
    pad     = torch.zeros(2,2,mh, device = device, dtype=torch.float32)
    y       = torch.cat((pad, Rx, pad), -1)
    y       /= torch.mean(y[:,0,:]**2 + y[:,1,:]**2 )

    out     = torch.zeros(2,2,N//sps, device = device)
    e       = torch.empty(N//sps, 2, device = device)
    buf     = torch.empty(2,2,2,N//sps, M, device = device)

    # downsampling included
    for i in torch.arange(mh,N+mh,sps):
        ind = torch.arange(-mh+i,i+mh+1)
        k   = i//sps - mh
        
        # Estimate Symbol 
        out[0,0,k]  = (
            torch.matmul(y[0,0,ind],h[0,0,0,:]) - torch.matmul(y[0,1,ind],h[0,0,1,:]) +
            torch.matmul(y[1,0,ind],h[0,1,0,:]) - torch.matmul(y[1,1,ind],h[0,1,1,:])
        )
        
        out[1,0,k]  = (
            torch.matmul(y[0,0,ind],h[1,0,0,:]) - torch.matmul(y[0,1,ind],h[1,0,1,:]) +
            torch.matmul(y[1,0,ind],h[1,1,0,:]) - torch.matmul(y[1,1,ind],h[1,1,1,:])
        )
        
        # Estimate Symbol 
        out[0,1,k]  = (
            torch.matmul(y[0,0,ind],h[0,0,1,:]) + torch.matmul(y[0,1,ind],h[0,0,0,:]) +
            torch.matmul(y[1,0,ind],h[0,1,1,:]) + torch.matmul(y[1,1,ind],h[0,1,0,:])
        )
        
        out[1,1,k]  = (
            torch.matmul(y[0,0,ind],h[1,0,1,:]) + torch.matmul(y[0,1,ind],h[1,0,0,:]) +
            torch.matmul(y[1,0,ind],h[1,1,1,:]) + torch.matmul(y[1,1,ind],h[1,1,0,:])
        )

        # Calculate error
        e[k,0]      = R - out[0,0,k]**2 - out[0,1,k]**2
        e[k,1]      = R - out[1,0,k]**2 - out[1,1,k]**2

        if eval == True:
            # buffering the filter update increments 
            buf[0,0,0,k,:] = out[0,0,k]*y[0,0,ind] + out[0,1,k]*y[0,1,ind]
            buf[0,0,1,k,:] = out[0,1,k]*y[0,0,ind] - out[0,0,k]*y[0,1,ind]
            buf[0,1,0,k,:] = out[0,0,k]*y[1,0,ind] + out[0,1,k]*y[1,1,ind]
            buf[0,1,1,k,:] = out[0,1,k]*y[1,0,ind] - out[0,0,k]*y[1,1,ind]

            buf[1,0,0,k,:] = out[1,0,k]*y[0,0,ind] + out[1,1,k]*y[0,1,ind] 
            buf[1,0,1,k,:] = out[1,1,k]*y[0,0,ind] - out[1,0,k]*y[0,1,ind]
            buf[1,1,0,k,:] = out[1,0,k]*y[1,0,ind] + out[1,1,k]*y[1,1,ind]
            buf[1,1,1,k,:] = out[1,1,k]*y[1,0,ind] - out[1,0,k]*y[1,1,ind]      

            # batch-wise updating with flexible step length
            if (k%symb_step == 0 and k>=NSymbBatch):
                
                # Update filters
                h[0,0,0,:] += 2*lr*torch.sum(torch.mul(buf[0,0,0,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,0]),dim=1)
                h[0,0,1,:] += 2*lr*torch.sum(torch.mul(buf[0,0,1,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,0]),dim=1) 
                h[0,1,0,:] += 2*lr*torch.sum(torch.mul(buf[0,1,0,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,0]),dim=1) 
                h[0,1,1,:] += 2*lr*torch.sum(torch.mul(buf[0,1,1,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,0]),dim=1) 

                h[1,0,0,:] += 2*lr*torch.sum(torch.mul(buf[1,0,0,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,1]),dim=1) 
                h[1,0,1,:] += 2*lr*torch.sum(torch.mul(buf[1,0,1,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,1]),dim=1) 
                h[1,1,0,:] += 2*lr*torch.sum(torch.mul(buf[1,1,0,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,1]),dim=1) 
                h[1,1,1,:] += 2*lr*torch.sum(torch.mul(buf[1,1,1,k-NSymbBatch:k,:].T,e[k-NSymbBatch:k,1]),dim=1) 
    return out, h, e


#%% [C3]
def compute_vae_loss(tx,rx):
    
    # q.size [2,8,NSbB]
    q       = rx["minibatch_output"].squeeze()
    tmp     = rx["minibatch_real"].squeeze()
    
    rxsig      = torch.zeros((2,2,rx['NsampBatch']))
    rxsig[0,0] = tmp[0]
    rxsig[0,1] = tmp[1]
    rxsig[1,0] = tmp[2]
    rxsig[1,1] = tmp[3]
    
    del tmp
    
    h       = rx["h_est"]      
    h_absq  = torch.sum(h**2, dim=2)
        
    mh      = tx["NSymbTaps"]-1
    Mh      = 2*mh
    
    Eq      = misc.my_zeros_tensor((tx["Npolars"],2,rx["NsampBatch"]))
    Var     = misc.my_zeros_tensor((tx["Npolars"],2,rx["NsampBatch"]))
    
    # compute expectation (with respect to q) of x and x**2
    # size [2,8,NSbB]
    amps_mat= tx["amps"].repeat(tx["Npolars"],rx["NSymbBatch"],2).transpose(1,2)
    
    xc_0    = (amps_mat * q)[:,:tx["N_amps"],:]         # size [2,4,NSbB]
    xc_1    = (amps_mat * q)[:,tx["N_amps"]:,:]         # size [2,4 NSbB]
    
    xc2_0   = ((amps_mat**2) * q)[:,:tx["N_amps"],:]    # size [2,4 NSbB]
    xc2_1   = ((amps_mat**2) * q)[:,tx["N_amps"]:,:]    # size [2,4 NSbB]
    
    # xco_<*> == 2*4*NSbB multiplications + 2*4*NSbB affectations
    # xc2_<*> == (2*4*NSbB)^2 multiplcations + 2*4*NSbB affectations
    
# =============================================================================
#     # xc<*> =====> (2*4*NSbB) * 2 multiplications = 16*NSbB
#     # xc2<*> =====> [(2*4*NSbB)^2] * 2 multiplications = 64*NSbB^2
#
#     # xc<*> =====> (2*4*NSbB) * 2 affectations == 16*NSbB affectations
#     # xc2<*> =====> (2*4*NSbB) * 2 affectations == 16*NSbB affectations
# =============================================================================

    Eq[:,0,::tx["Nsps"]]    = torch.sum(xc_0, dim=1)    # size [2,NSbB]
    Eq[:,1,::tx["Nsps"]]    = torch.sum(xc_1, dim=1)    # size [2,NSbB]
    
    Var[:,0,::tx["Nsps"]]   = torch.sum(xc2_0, dim=1)   # size [2,NSbB]
    Var[:,1,::tx["Nsps"]]   = torch.sum(xc2_1, dim=1)   # size [2,NSbB]
    
    # sum(xc_<*>)     == 2*NSbB summations + 2*NSbB affectations
    # sum(xc2_<*>)    == 2*NSbB summations + 2*NSbB affectations
    # 
    # Eq            == 2*(2*NSbB) summations + 2*(2*NSbB) affectations
    # Var           == 2*(2*NSbB) summations + 2*(2*NSbB) affectations

    Var                     = Var - Eq**2

    # Eq**2         == [2*2*NspB]**2 multiplications = [4*(2NSbB-1)]^2= 64*NSbB^2-16
    # Var-Eq**2     == 2*2*NspB summations  = 4*(2*NSbB-1) = 8NSbB - 4
    # Var           == 2*2*NspB affectations = 8NSbB - 4
    
# =============================================================================
#     # Var =======>    16*NSbB-4 Summations,
#                       64*NSbB^2-16 NSbB multiplications,
#                       16*NSbB-4 affectations
# =============================================================================
    
    D_real  = misc.my_zeros_tensor((2,rx["NsampBatch"]-Mh))  
    D_imag  = misc.my_zeros_tensor((2,rx["NsampBatch"]-Mh))
    Etmp    = misc.my_zeros_tensor((2))
    idx     = np.arange(Mh,rx["NsampBatch"])
    nm      = idx.shape[0]

    for j in range(Mh+1): # h[chi,nu,c,k]
        D_real  = D_real + h[:,0,0:1,j].expand(-1,nm) * Eq[0,0:1,idx-j].expand(tx["Npolars"],-1)\
                         - h[:,0,1:2,j].expand(-1,nm) * Eq[0,1:2,idx-j].expand(tx["Npolars"],-1)\
                         + h[:,1,0:1,j].expand(-1,nm) * Eq[1,0:1,idx-j].expand(tx["Npolars"],-1)\
                         - h[:,1,1:2,j].expand(-1,nm) * Eq[1,1:2,idx-j].expand(tx["Npolars"],-1)

        
        D_imag  = D_imag + h[:,0,1:2,j].expand(-1,nm) * Eq[0,0:1,idx-j].expand(tx["Npolars"],-1)\
                         + h[:,0,0:1,j].expand(-1,nm) * Eq[0,1:2,idx-j].expand(tx["Npolars"],-1)\
                         + h[:,1,1:2,j].expand(-1,nm) * Eq[1,0:1,idx-j].expand(tx["Npolars"],-1)\
                         + h[:,1,0:1,j].expand(-1,nm) * Eq[1,1:2,idx-j].expand(tx["Npolars"],-1)


# =============================================================================
#         # x = NspB-NspT = 2*NSbB - 2*NSbT -1
# =============================================================================
        #  h<*>.* Eq[0,0:1,idx-j].expand(tx["Npolars"],-1) ==
        #           2*x multiplications
        # D_<*> + <*>   == 4*{2*x} summations + 4*2*x multiplications
        # D_<*> =       == 2*x affectations
        
        
        Var_sum = torch.sum(Var[:,:,idx-j], dim=(1,2))        
        Etmp    = Etmp+h_absq[:,0,j] * Var_sum[0] + h_absq[:,1,j] * Var_sum[1]
        # Var_sum == 2*2*x summations + 1 affectation
        # Etmp    == 2*2 multiplications + 2 summations + 1 affectation
   
    
# =============================================================================
#           | S             | M                 | A
# Dreal     | 4x*Mh         | 8*x*Mh            | 2x*Mh
# Dimag     | 4x*Mh         | 8*x*Mh            | 2x*Mh
# VarSUm    | 4x*Mh         |                   | 1*Mh
# Etmp      | 2x*Mh         | 4*x*Mh            | 1*Mh
#            O(NSbT*NSbB)       O(NSbT*NSbB)        O(NSbT*NSbB)
# =============================================================================

    TT          = tx["prob_amps"].repeat(rx["NSymbBatch"]-Mh,2).transpose(0,1)   # P(x)

    ynorm2      = torch.sum(rxsig[:,:,mh:-mh]**2, dim=(1,2))
    
    # (2*2*x)^2 mulitplications + 2*2*x summations + 2*2*x affectations

    DInorm2     = torch.sum(D_real**2, dim=1)
    DQnorm2     = torch.sum(D_imag**2, dim=1)
    # D<*>  == (2*x)^2 multiplications + x summations + 2 affectations
    E           = DInorm2+DQnorm2+Etmp
    # E == 3*2 sommations + 2 affectations
    
    yIT         = rxsig[:,0,mh:-mh]
    yQT         = rxsig[:,1,mh:-mh]
    C           = ynorm2-2*torch.sum(yIT*D_real+yQT*D_imag,dim=1)+E
    # y<*>*D_<*> == 2*[2*(2*x)] mulitplications + 2*x+2+2 summations + 2 affectations
    
    Llikelihood = torch.sum((rx["NsampBatch"]-Mh)*torch.log(C))
    # 2 multiplications + 1 summation + 1 affectation
    DKL         = torch.sum(q[0,:,mh:-mh]*torch.log(q[0,:,mh:-mh]/TT+ 1e-12) \
                           +q[1,:,mh:-mh]*torch.log(q[1,:,mh:-mh]/TT+ 1e-12) )
    #  q[0,:,mh:-mh]*torch.log(q[0,:,mh:-mh]/TT+ 1e-12) == 2*8*NSbB multiplications+ 8*NSbB summations
    # DKL == 2*(8*NSbB) summations

    loss        = Llikelihood - 1*DKL 
    Pnoise_batch= (C/(rx["NsampBatch"]-Mh)).detach()

# =============================================================================
#           | S             | M                         | A
# ynorm2    | 4x            | 16*x^2                    | 4x 
# DInorm2   | x             | 4x^2                      | 2
# DQnorm2   | x             | 4x^2                      | 2
# E         | 6             |                           | 2
# C         | 2x+4          | 8x                        | 2
# Llike     | 1             | 2                         | 1
# DKL       | 16x           | 32*x                      | 2
# loss      | 1             |                           | 1
# Pnoise_   | 1             | 1                         | 1
#             24x = O(NSbB)  24x^2+40x = O(NSbB^2)         4x = O(NSbB)
# =============================================================================


    rx['losses_subframe'][rx["Frame"]][rx['BatchNo']]       = loss.item()
    rx['DKL_subframe'][rx["Frame"]][rx['BatchNo']]          = DKL.item()
    rx['Llikelihood_subframe'][rx["Frame"]][rx['BatchNo']]  = Llikelihood.item()
    rx['Pnoise_batches'][:,rx['BatchNo']]                   = Pnoise_batch
    
    
# =============================================================================
#           | S             | M                         | A
#            O(NSbT*NSbB)       O(NSbT*NSbB)                O(NSbT*NSbB)
#            O(NSbB)            O(NSbB^2)                   O(NSbB)
# -----------------------------------------------------------------------------
#           O(NSbT*NSbB)        O(NSbB^2)                   O(NSbT*NSbB)
# =============================================================================
    return rx,loss


#%%
def CPE(rx): # Carrier Phase Estimation (Viterbi-Viterbi)
    
    if rx['mimo'].lower() == "cma" and rx['Frame'] >= rx['FrameChannel']:
        y               = rx['out_mimo'][str(rx['Frame']- rx['FrameChannel'])]
        pi              = torch.tensor(3.141592653589793)
        pi2             = pi/2
        pi4             = pi/4
        M_ma            = 101     # length of moving average filter 
        y_corr          = torch.zeros_like(y)
        y_pow4          = torch.zeros_like(y)
        
        ax              = y[0,0,:]
        bx              = y[0,1,:]
        ay              = y[1,0,:] 
        by              = y[1,1,:]
        
        # taking the signal to the 4th power to cancel out modulation
        # # (a+jb)^4 = a^4 - 6a^2b^2 + b^4 +j(4a^3b - 4ab^3)
        ax2             = ax**2
        bx2             = bx**2
        ay2             = ay**2
        by2             = by**2
        
        y_pow4[0,0,:]   = ax2*ax2 - torch.full_like(ax,6)*ax2*bx2 + bx2*bx2 
        y_pow4[0,1,:]   = torch.full_like(ax,4)*(ax2*ax*bx - ax*bx2*bx)
        y_pow4[1,0,:]   = ay2*ay2 - torch.full_like(ay,6)*ay2*by2 + by2*by2 
        y_pow4[1,1,:]   = torch.full_like(ay,4)*(ay2*ay*by - ay*by2*by)
        
        # moving average filtering
        kernel_ma       = torch.full((1,1,M_ma), 1/M_ma, device=y.device, dtype=torch.float32)
        y_conv          = misc.my_zeros_tensor((4,1,y_pow4.shape[2]))
        
        y_conv[0,0,:]   = y_pow4[0,0,:]
        y_conv[1,0,:]   = y_pow4[0,1,:]
        y_conv[2,0,:]   = y_pow4[1,0,:]
        y_conv[3,0,:]   = y_pow4[1,1,:]
    
        y_ma            = F.conv1d(y_conv,kernel_ma,bias=None,padding=M_ma//2)
    
        phiX_corr       = torch.atan2(y_ma[1,0,:],-y_ma[0,0,:])/4
        diff_phiX       = phiX_corr[1:] - phiX_corr[:-1]
    
        ind_X_pos       = torch.nonzero(diff_phiX>pi4)
        ind_X_neg       = torch.nonzero(diff_phiX<-pi4)
    
        for i in ind_X_pos:     # unwrapping
            phiX_corr[i+1:] -=  pi2
        for j in ind_X_neg:
            phiX_corr[j+1:] +=  pi2
    
        cos_phiX        = torch.cos(phiX_corr)
        sin_phiX        = torch.sin(phiX_corr)
    
        phiY_corr       = torch.atan2(y_ma[3,0,:],-y_ma[2,0,:])/4
        diff_phiY       = phiY_corr[1:] - phiY_corr[:-1]
    
        ind_Y_pos       = torch.nonzero(diff_phiY>pi4)
        ind_Y_neg       = torch.nonzero(diff_phiY<-pi4)
        
        for ii in ind_Y_pos:    # unwrapping 
            phiY_corr[ii+1:] -=  pi2
        for jj in ind_Y_neg:
            phiY_corr[jj+1:] +=  pi2
    
        cos_phiY        = torch.cos(phiY_corr)
        sin_phiY        = torch.sin(phiY_corr)
    
        # compensating phase offset
        y_corr[0,0,:]   = ax*cos_phiX - bx*sin_phiX
        y_corr[0,1,:]   = bx*cos_phiX + ax*sin_phiX
        y_corr[1,0,:]   = ay*cos_phiY - by*sin_phiY
        y_corr[1,1,:]   = by*cos_phiY + ay*sin_phiY
        
        rx['out_cpe'][str(rx['Frame']- rx['FrameChannel'])] = y_corr
    return rx





#%%
# ============================================================================================ #
# hard decision based on the decision boundaries d_vec0 (lower) and d_vec1 (upper)
# ============================================================================================ #
def dec_on_bound(rx,tx_int,d_vec0, d_vec1):
    
    SER     = torch.zeros(rx.shape[0], dtype = torch.float32, device = rx.device)
    
    xI0     = d_vec0.index_select(dim=0, index=tx_int[0,0,:])
    xI1     = d_vec1.index_select(dim=0, index=tx_int[0,0,:])
    corr_xI = torch.bitwise_and((xI0 <= rx[0, 0, :]), (rx[0, 0, :] < xI1))
    
    xQ0     = d_vec0.index_select(dim=0, index=tx_int[0,1,:])
    xQ1     = d_vec1.index_select(dim=0, index=tx_int[0,1,:])
    corr_xQ = torch.bitwise_and((xQ0 <= rx[0, 1, :]), (rx[0, 1, :] < xQ1))

    yI0     = d_vec0.index_select(dim=0, index=tx_int[1,0,:])
    yI1     = d_vec1.index_select(dim=0, index=tx_int[1,0,:])
    corr_yI = torch.bitwise_and((yI0 <= rx[1, 0, :]), (rx[1, 0, :] < yI1))

    yQ0     = d_vec0.index_select(dim=0, index=tx_int[1,1,:])
    yQ1     = d_vec1.index_select(dim=0, index=tx_int[1,1,:])
    corr_yQ = torch.bitwise_and((yQ0 <= rx[1, 1, :]), (rx[1, 1, :] < yQ1))

    # no error only if both I or Q are correct
    ex      = ~(torch.bitwise_and(corr_xI,corr_xQ))
    ey      = ~(torch.bitwise_and(corr_yI,corr_yQ))
    
    # SER = numb. of errors/ num of symbols
    SER[0]  = torch.sum(ex)/ex.nelement()
    SER[1]  = torch.sum(ey)/ey.nelement()
    return SER

#%%
# ============================================================================================ #
# find shiftings in both Npolarsarisation and time by correlation with expectation of x^I with respect to q
# ============================================================================================ #   

def find_shift(tx,rx):

    
    corr_max    = misc.my_zeros_tensor((2,2,2))
    corr_ind    = misc.my_zeros_tensor((2,2,2))
    
    # amplitudes of the constellation
    amTT        = tx['amps'].repeat(tx['Npolars'],rx['NSymbFrame'],1).transpose(1,2)
    
    if rx['mimo'].lower() == "vae":
        # calculate expectation E_q(x^I) of in-phase component
        E       = torch.sum(amTT *  rx['out_train'][:,:tx['N_amps']], dim=1)
    else:
        E       = rx["out_train"][:,0]
        
    # correlate with (both Npolarsarisations) and shifted versions in time --> find max. correlation
    E_mat = misc.my_zeros_tensor((2,rx['NSymbFrame'],rx['NSymbCut_tot']))
 
    for k in range(rx['NSymbCut_tot']):
        E_mat[:,:,k] = torch.roll(E,k-rx["NSymbCut"],-1)
        
    xcorr0 = tx['Symb_real'][:,0].float() @ E_mat
    xcorr1 = tx['Symb_real'][:,1].float() @ E_mat
    
    corr_max[0,:,:] , corr_ind[0,:,:]   = torch.max(torch.abs(xcorr0), dim=-1)
    corr_max[1,:,:] , corr_ind[1,:,:]   = torch.max(torch.abs(xcorr1), dim=-1) 
    corr_max        , ind_max           = torch.max(corr_max,dim=0); #corr_ind = corr_ind[ind_max]
    
    ind_XY      = misc.my_zeros_tensor(2,dtype=torch.int16)
    ind_YX      = misc.my_zeros_tensor(2,dtype=torch.int16)

    
    ind_XY[0]   = corr_ind[ind_max[0,0],0,0]
    ind_XY[1]   = corr_ind[ind_max[1,1],1,1]
    ind_YX[0]   = corr_ind[ind_max[0,1],0,1]
    ind_YX[1]   = corr_ind[ind_max[1,0],1,0] 

    if (corr_max[0,0]+corr_max[1,1]) >= (corr_max[0,1]+corr_max[1,0]):
        return rx["NSymbCut"]-ind_XY, 0

    else:
        return rx["NSymbCut"]-ind_YX, 1


    
#%%
# ============================================================================================ #
# find shiftings in both Npolarsarisation and time by correlation with the constellation output's in-phase component x^I 
# ============================================================================================ #
def find_shift_symb_full(rx, tx, N_shift): 
    corr_max = torch.empty(2,2,2, device = rx.device, dtype=torch.float32)
    corr_ind = torch.empty_like(corr_max)
    len_corr = rx.shape[-1] #torch.max(q.shape[-1],1000)
    E = rx[:,0,:len_corr] 

    # correlate with (both Npolarsarisations) and shifted versions in time --> find max. correlation
    E_mat = torch.empty(2,len_corr,N_shift, device=rx.device, dtype=torch.float32)
    for i in range(N_shift):
        E_mat[:,:,i] = torch.roll(E,i-N_shift//2,-1)
    corr_max[0,:,:], corr_ind[0,:,:] = torch.max( torch.abs(tx[:,0,:len_corr].float() @ E_mat), dim=-1)
    corr_max[1,:,:], corr_ind[1,:,:] = torch.max( torch.abs(tx[:,1,:len_corr].float() @ E_mat), dim=-1)
    corr_max, ind_max = torch.max(corr_max,dim=0); 

    ind_XY = torch.zeros(2,device=rx.device, dtype=torch.int16); ind_YX = torch.zeros_like(ind_XY)
    ind_XY[0] = corr_ind[ind_max[0,0],0,0]; ind_XY[1] = corr_ind[ind_max[1,1],1,1]
    ind_YX[0] = corr_ind[ind_max[0,1],0,1]; ind_YX[1] = corr_ind[ind_max[1,0],1,0] 
    
    if (corr_max[0,0]+corr_max[1,1]) >= (corr_max[0,1]+corr_max[1,0]):
        return N_shift//2-ind_XY, 0
    else:
        return N_shift//2-ind_YX, 1
    


#%%
def SER_estimation(tx,rx):
    

    if rx['mimo'].lower() == "vae":
        shift,r                 = find_shift(tx,rx)

    else:
        if rx["Frame"]< rx['FrameChannel']:
            rx["SER_valid"][2:,rx['Frame']] = np.nan
            rx["shift"]                     = torch.tensor([float("nan"),float("nan")])
            rx["r"]                         = torch.tensor([float("nan")])
            return rx
        else:
            # indexMax        = np.shape(rx['out_cpe'][str(rx['Frame']-rx['FrameChannel'])])[-1]
            # sig_eq          = rx['out_cpe'][str(rx['Frame']-rx['FrameChannel'])]
            shift,r         = find_shift(tx,rx)
            
            
    rx['NSymbSER']      = rx['NSymbBatch'] - shift[0] - rx["NSymbCut"]
    # compensate Npolars. shift
    out_train               = rx['out_train'].roll(r,0)
    
    # compensate time shift (in multiple symb.)
    out_train[0,:,:]        = out_train[0,:,:].roll(int(-shift[0]),-1)
    out_train[1,:,:]        = out_train[1,:,:].roll(int(-shift[1]),-1)
    
    temp_out_train          = torch.reshape(out_train,(tx["Npolars"],2*tx["N_amps"],rx["NBatchFrame"],rx['NSymbBatch']))
    
    # cut off edge symbols to avoid edge effects
    temp_out_train_SER      = temp_out_train[:,:,:,:rx['NSymbSER']]
    temp_out_train          = torch.reshape(temp_out_train_SER,(tx["Npolars"],2*tx["N_amps"],-1))


    symb_real               = torch.reshape(tx["Symb_real"],(tx["Npolars"],2,rx["NBatchFrame"],rx['NSymbBatch']))
    symb_real_SER           = symb_real[:, :, :, :rx['NSymbSER']]
    temp_data_tensor        = torch.reshape(symb_real_SER,(tx["Npolars"], 2, -1))
    
    rx["SER_valid"][:,rx['Frame']]     = SER_IQflip(temp_out_train[:,:,11:-11-torch.max(torch.abs(shift))],
                                              temp_data_tensor[:,:,11:-11-torch.max(torch.abs(shift))])

    rx['shift'] = shift
    rx['r']     = r
    
    tx      = misc.sort_dict_by_keys(tx)
    rx      = misc.sort_dict_by_keys(rx)
    

    
    return rx





#%%
# ============================================================================================ #
# estimate symbol error rate from estimated a posterioris q
# ============================================================================================ #
def SER_IQflip(q, tx): 
    
    N_amp               = q.shape[1]//2
    dec                 = torch.empty_like(tx, dtype=torch.int16)
    data                = torch.empty_like(tx, dtype=torch.int16)
    data_IQinv          = torch.empty_like(data)
    SER                 = torch.ones(2,2,4)
    
    scale               = (N_amp-1)/2
    data                = torch.round(scale*tx.float()+scale) # decode TX
    data_IQinv[:,0,:]   = data[:,0,:]
    data_IQinv[:,1,:]   = -(data[:,1,:]-scale*2)  # compensate potential IQ flip
    
    
    ### zero phase-shift
    dec[:,0,:]          = torch.argmax(q[:,:N_amp,:], dim=1)
    dec[:,1,:]          = torch.argmax(q[:,N_amp:,:], dim=1) # hard decision on max(q)
    
    SER[0,:,0]          = torch.mean( ((data - dec).bool().any(dim=1)).to(torch.float), dim=-1)
    SER[1,:,0]          = torch.mean( ((data_IQinv - dec).bool().any(dim=1)).to(torch.float), dim=-1)
    
    ### pi phase-shift
    dec_pi              = -(dec-scale*2)
    SER[0,:,1]          = torch.mean( ((data - dec_pi).bool().any(dim=1)).to(torch.float), dim=-1)
    SER[1,:,1]          = torch.mean( ((data_IQinv - dec_pi).bool().any(dim=1)).to(torch.float), dim=-1)

    ### pi/4 phase-shift
    dec_pi4             = torch.empty_like(dec)
    dec_pi4[:,0,:]      = -(dec[:,1,:]-scale*2)
    dec_pi4[:,1,:]      = dec[:,0,:]
    SER[0,:,2]          = torch.mean( ((data - dec_pi4).bool().any(dim=1)).to(torch.float), dim=-1)
    SER[1,:,2]          = torch.mean( ((data_IQinv - dec_pi4).bool().any(dim=1)).to(torch.float), dim=-1)

    ### 3pi/4 phase-shift
    dec_3pi4            = -(dec_pi4-scale*2)
    SER[0,:,3]          = torch.mean( ((data - dec_3pi4).bool().any(dim=1)).to(torch.float), dim=-1)
    SER[1,:,3]          = torch.mean( ((data_IQinv - dec_3pi4).bool().any(dim=1)).to(torch.float), dim=-1)

    SER_out             = torch.amin(SER, dim=(0,-1))   # choose minimum estimation per Npolarsarisation
    
    return SER_out 

#%%
# ============================================================================================ #
# estimate symbol error rate from output constellation by considering PCS
# ============================================================================================ #
def SER_constell_shaping(rx, tx, amp_levels, nu_sc, var): 

    device = rx.device
    N_amp = amp_levels.shape[0]
    data = torch.empty_like(tx, device=device, dtype=torch.int32)
    data_IQinv = torch.empty_like(data)
    SER = torch.ones(2,2,4, device=device)

    # calculate decision boundaries based on PCS

    d_vec = (1+2*nu_sc*var[0])*(amp_levels[:-1] + amp_levels[1:])/2
    d_vec0 = torch.cat(((-Inf*torch.ones(1).to(device)),d_vec),dim=0)
    d_vec1 = torch.cat((d_vec,Inf*torch.ones(1).to(device)))
    
    scale = (N_amp-1)/2
    data = torch.round(scale*tx.float()+scale).to(torch.int32) # decode TX
    data_IQinv[:,0,:], data_IQinv[:,1,:] = data[:,0,:], -(data[:,1,:]-scale*2)  # compensate potential IQ flip

    rx *= torch.mean(torch.sqrt(tx[:,0,:].float()**2 + tx[:,1,:].float()**2)) /torch.mean(torch.sqrt(rx[:,0,:]**2 + rx[:,1,:]**2)) # normalize constellation output

    ### zero phase-shift  torch.sqrt(2*torch.mean(rx[0,:N*sps:sps]**2))
    SER[0,:,0] = dec_on_bound(rx,data,d_vec0, d_vec1)
    SER[1,:,0] = dec_on_bound(rx,data_IQinv,d_vec0, d_vec1)
    
    ### pi phase-shift
    rx_pi = -(rx).detach().clone()
    SER[0,:,1] = dec_on_bound(rx_pi,data,d_vec0, d_vec1)
    SER[1,:,1] = dec_on_bound(rx_pi,data_IQinv,d_vec0, d_vec1)

    ### pi/4 phase-shift
    rx_pi4 = torch.empty_like(rx)
    rx_pi4[:,0,:], rx_pi4[:,1,:] = -(rx[:,1,:]).detach().clone(), rx[:,0,:]
    SER[0,:,2] = dec_on_bound(rx_pi4,data,d_vec0, d_vec1)
    SER[1,:,2] = dec_on_bound(rx_pi4,data_IQinv,d_vec0, d_vec1)

    ### 3pi/4 phase-shift
    rx_3pi4 = -(rx_pi4).detach().clone()
    SER[0,:,3] = dec_on_bound(rx_3pi4,data,d_vec0, d_vec1)
    SER[1,:,3] = dec_on_bound(rx_3pi4,data_IQinv,d_vec0, d_vec1)

    SER_out = torch.amin(SER, dim=(0,-1))       # choose minimum estimation per Npolarsarisation
    return SER_out 







#%%
# ============================================================================================ #
# Soft demapping with correction term for PCS:  + nu_sc * amp_levels**2 --
# see SD-FEC in Junho Cho, "Probabilistic Constellation Shaping for OpticalFiber Communications"
# ============================================================================================ #
def soft_dec(out, var, amp_levels, nu_sc): 
    
    dev             = out.device
    dty             = out.dtype
    
    N               = out.shape[-1]
    n               = amp_levels.shape[0]
    
    q_est           = torch.empty(2,2*n,N, device=dev, dtype=dty)
    amp_lev_mat     = amp_levels.repeat(N,1).transpose(0,1)
    amp_lev_mat_sq  = amp_lev_mat**2
    
    out_I           = out[:,0,:]
    out_Q           = out[:,1,:]

    sm              = nn.Softmin(dim=0)
    
    q_est[0,:n,:]   = sm( (out_I[0,:]-amp_lev_mat)**2/2/var[0] + nu_sc * amp_lev_mat_sq)
    q_est[0,n:,:]   = sm( (out_Q[0,:]-amp_lev_mat)**2/2/var[0] + nu_sc * amp_lev_mat_sq)
    q_est[1,:n,:]   = sm( (out_I[1,:]-amp_lev_mat)**2/2/var[1] + nu_sc * amp_lev_mat_sq)
    q_est[1,n:,:]   = sm( (out_Q[1,:]-amp_lev_mat)**2/2/var[1] + nu_sc * amp_lev_mat_sq)
    
    return q_est


#%% [C3]
class twoXtwoFIR(nn.Module):
    
    # caution: rx['noise_var'] will be missing if rxhw.load_ase is off
    # initialisation of tw0XtwoFIR with nn.Module.__init__ method
    # but with the attributes of tw0XtwoFIR
    def __init__(self, tx):
        
        super(twoXtwoFIR, self).__init__()
        
        self.conv_w = nn.Conv1d(
            in_channels     = 4,
            out_channels    = 2,
            kernel_size     = tx['NsampTaps'],
            bias            = False,
            padding         = tx['NsampTaps']//2,
            stride          = tx["Nsps"]
            ).to(dtype=torch.float32)
        
        # Dirac-initilisation
        nn.init.dirac_(self.conv_w.weight)

        # soft demapper -- softmin includes minus in exponent
        self.sm = nn.Softmin(dim=0)
                                   
    # calculates q and the estimated data x (before the soft demapper), (see [1])
    def forward(self, tx,rx): 

        # does the convolution between the FIR filter and the estimated data x
        # (used for x=received signal divided in batches in the processing function)
        # = equalization
        #
        # self = the FIR filter that is optimized at each frame
        # out_j = self.conv_w(rx_sig_j) where j == {I,Q}
        # rx_sig_j.shape = (Npolars,2,NsampBatch=sps*NSymbBatch)

        n               = tx["amps"].shape[0]
        q_est           = misc.my_zeros_tensor((2,2*n,rx["NSymbBatch"]))
        amp_lev_mat     = tx["amps"].repeat(rx["NSymbBatch"],1).transpose(0,1)
        amp_lev_mat_sq  = amp_lev_mat**2
        
        if type(rx["minibatch_real"]) != torch.Tensor:
            rx["minibatch_real"] = torch.tensor(rx["minibatch_real"])
            
        # NsampBatch      = rx["minibatch_real"][:,0].shape[1]  
        YHI             = rx["minibatch_real"][0].view(1,rx['NsampBatch'])
        YHQ             = rx["minibatch_real"][1].view(1,rx['NsampBatch'])
        YVI             = rx["minibatch_real"][2].view(1,rx['NsampBatch'])
        YVQ             = rx["minibatch_real"][3].view(1,rx['NsampBatch'])

        in_I            = torch.cat((YHI,YVI,-YHQ,-YVQ),0)
        in_Q            = torch.cat((YHQ,YVQ,YHI,YVI),0)
            
        out_I           = self.conv_w(in_I)
        out_Q           = self.conv_w(in_Q)

        ZHI             = out_I[0,:]
        ZVI             = out_I[1,:]
        ZHQ             = out_Q[0,:]
        ZVQ             = out_Q[1,:]
        
        rx["sig_mimo_real"][0,rx['Frame'],rx['BatchNo'],:] = ZHI.detach().numpy()
        rx["sig_mimo_real"][1,rx['Frame'],rx['BatchNo'],:] = ZHQ.detach().numpy()
        rx["sig_mimo_real"][2,rx['Frame'],rx['BatchNo'],:] = ZVI.detach().numpy()
        rx["sig_mimo_real"][3,rx['Frame'],rx['BatchNo'],:] = ZVQ.detach().numpy()
        
        # Soft demapping
        # correction term for PCS: + nu_sc * amp_levels**2 -- see [2]
        # calculation of q according to the paper, with the comparison of the
        # estimated x_hat
        # (after equalizer, before soft demapper) to all the possible
        # amplitudes
        
        q_est[0, :n, :] = self.sm((ZHI-amp_lev_mat)**2/2/rx["noise_var"][0]
                                          +  tx["nu_sc"]* amp_lev_mat_sq)
        
        q_est[0, n:, :] = self.sm((ZHQ-amp_lev_mat)**2/2/rx["noise_var"][0]
                                          +  tx["nu_sc"]* amp_lev_mat_sq)
        
        q_est[1,:n,:]   = self.sm((ZVI-amp_lev_mat)**2/2/rx["noise_var"][1]
                                          +  tx["nu_sc"]* amp_lev_mat_sq)
        
        q_est[1,n:,:]   = self.sm((ZVQ-amp_lev_mat)**2/2/rx["noise_var"][1]
                                          +  tx["nu_sc"] * amp_lev_mat_sq)
        

        rx['minibatch_output'] = q_est
        
        return rx


#%%
            
def train_vae(BatchNo,rx,tx):
    
    rx['BatchNo']           = BatchNo   
    rx["minibatch_real"]    = rx["sig_real"][:,BatchNo*rx["NsampBatch"]:(BatchNo+1)*rx["NsampBatch"]]
    rx                      = rx['net'](tx,rx)
    
    rx['out_train'][:,:,BatchNo*rx["NSymbBatch"]:(BatchNo+1)*rx["NSymbBatch"]] = \
                            rx['minibatch_output'].detach().clone()

    rx['out_train'][:,:,BatchNo*rx["NSymbBatch"]:(BatchNo+1)*rx["NSymbBatch"]] = \
                            rx['minibatch_output'].detach().clone()
    
    return rx







#%%