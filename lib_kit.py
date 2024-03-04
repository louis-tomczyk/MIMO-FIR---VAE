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
from numpy.core.numeric import Inf
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib_misc as misc
import lib_general as gen
import lib_kit as kit

pi = np.pi

# =============================================================================
# Contents
# =============================================================================
# loss_function_shaping     (tx,rx)
# twoXtwoFIR                (nn.Module)
# train_self                (batchNo,rx,tx)
# train_ext                 (batch,rx,net,tx,out_train,out_const,data)
# init_dict                 ()
# CPE                       (y)
# SER_IQflip                (q, tx)
# SER_constell_shaping      (rx, tx, amp_levels, nu_sc, var)
# dec_on_bound              (rx,tx_int,d_vec0, d_vec1)
# find_shift                (q, tx, N_shift, amp_levels, Npolars)
# find_shift_symb_full      (rx, tx, N_shift)
# CMA                       (Rx, R, h, lr, sps, eval)
# CMAbatch                  (Rx, R, h, lr, batchlen, sps, eval)
# CMAflex                   (Rx, R, h, lr, batchlen, symb_step, sps, eval)
# soft_dec                  (out, var, amp_levels, nu_sc)
# =============================================================================

    
#%%
# =============================================================================
# computation of the loss function according to the article
# =============================================================================

def loss_function_shaping(tx,rx):
    
    q       = rx["minibatch_out"].squeeze()
    rxsig   = rx["minibatch_real"].squeeze()
    h       = rx["h_est"]      
    h_absq  = torch.sum(h**2, dim=2)
        
    mh      = tx["NSymbTaps"]-1
    Mh      = 2*mh
    
    Eq      = misc.my_zeros_tensor((tx["Npolars"],2,rx["NsampBatch"]))
    Var     = misc.my_zeros_tensor((tx["Npolars"],2,rx["NsampBatch"]))
    
    # compute expectation (with respect to q) of x and x**2
    amps_mat= tx["amps"].repeat(tx["Npolars"],rx["BatchLen"],2).transpose(1,2)
    
    xc_0    = (amps_mat * q)[:,:tx["N_amps"],:]
    xc_1    = (amps_mat * q)[:,tx["N_amps"]:,:]
    
    xc2_0   = ((amps_mat**2) * q)[:,:tx["N_amps"],:]
    xc2_1   = ((amps_mat**2) * q)[:,tx["N_amps"]:,:]

    Eq[:,0,::tx["Nsps"]]    = torch.sum(xc_0, dim=1)
    Eq[:,1,::tx["Nsps"]]    = torch.sum(xc_1, dim=1)
    Var[:,0,::tx["Nsps"]]   = torch.sum(xc2_0, dim=1)
    Var[:,1,::tx["Nsps"]]   = torch.sum(xc2_1, dim=1)
    Var                     = Var - Eq**2

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

        Var_sum = torch.sum(Var[:,:,idx-j], dim=(1,2))
        Etmp    = Etmp+h_absq[:,0,j] * Var_sum[0] + h_absq[:,1,j] * Var_sum[1]
    
    TT          = tx["prob_amps"].repeat(rx["BatchLen"]-Mh,2).transpose(0,1)   # P(x)

    # data_entropy = DKL
    # not really the entropy but the D_KL in the paper, equation 7 of arxiv
    DKL         = torch.sum(-q[0,:,mh:-mh]*torch.log(q[0,:,mh:-mh]/TT+ 1e-12) \
                           - q[1,:,mh:-mh]*torch.log(q[1,:,mh:-mh]/TT+ 1e-12) )

    ynorm2      = torch.sum(rxsig[:,:,mh:-mh]**2, dim=(1,2))
    yIT         = rxsig[:,0,mh:-mh]
    yQT         = rxsig[:,1,mh:-mh]
    DI          = D_real
    DQ          = D_imag
    DInorm2     = torch.sum(D_real**2, dim=1)
    DQnorm2     = torch.sum(D_imag**2, dim=1)    
    E           = DInorm2+DQnorm2+Etmp
    C           = ynorm2-2*torch.sum(yIT*DI+yQT*DQ,dim=1)+E
    
    loss        = torch.sum((rx["NsampBatch"]-Mh)*torch.log(C)) - 1*DKL 
    Pnoise_batch= (C/(rx["NsampBatch"]-Mh)).detach()


    rx['losses_subframe'][rx["Frame"]][rx['batchNo']]   = loss
    rx['Pnoise_batches'][:,rx['batchNo']]               = Pnoise_batch
                
                
    return rx,loss

#%%
# ============================================================================================ #
# complex-valued 2x2 butterfly FIR filter
# ============================================================================================ #
# initialises the filter with a dirac
# Bibliography: 
# [1] Caciularu et al. ICC 2018, 10.1109/ICCW.2018.8403666
# [2] SD-FEC in Junho Cho, "Probabilistic Constellation Shaping for OpticalFiber Communications"

class twoXtwoFIR(nn.Module):
    
    # initialisation of tw0XtwoFIR with nn.Module.__init__ method
    # but with the attributes of tw0XtwoFIR
    def __init__(self, tx):
        
        super(twoXtwoFIR, self).__init__()
        
        self.conv_w = nn.Conv1d(
            in_channels     = 4, # before it was 4
            out_channels    = 2,
            kernel_size     = tx["Ntaps"],
            bias            = False,
            padding         = tx["Ntaps"]//2,
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
        # rx_sig_j.shape = (Npolars,2,NsampBatch=sps*BatchLen)
        
        N   = rx["minibatch_real"][:,0,:].shape[1]  
        YHI = rx["minibatch_real"][0,0,:].view(1,N)
        YHQ = rx["minibatch_real"][0,1,:].view(1,N)
        YVI = rx["minibatch_real"][1,0,:].view(1,N)
        YVQ = rx["minibatch_real"][1,1,:].view(1,N)

        in_I            = torch.cat((YHI,YVI,-YHQ,-YVQ),0)
        in_Q            = torch.cat((YHQ,YVQ,YHI,YVI),0)
        
        out_I           = self.conv_w(in_I)
        out_Q           = self.conv_w(in_Q)

        n               = tx["amps"].shape[0]
        q_est           = misc.my_zeros_tensor((2,2*n,rx["BatchLen"]))
        out             = misc.my_zeros_tensor((2,2,rx["BatchLen"]))
        amp_lev_mat     = tx["amps"].repeat(rx["BatchLen"],1).transpose(0,1)
        amp_lev_mat_sq  = amp_lev_mat**2

        out[:,0,:]      = out_I
        out[:,1,:]      = out_Q

        # Soft demapping
        # correction term for PCS: + nu_sc * amp_levels**2 -- see [2]
        # calculation of q according to the paper, with the comparison of the estimated x_hat
        # (after equalizer, before soft demapper) to all the possible amplitudes 
        q_est[0, :n, :] = self.sm((out_I[0, :]-amp_lev_mat)**2/2/rx["noise_var"][0]
                                          +  tx["nu_sc"]* amp_lev_mat_sq)
        
        q_est[0, n:, :] = self.sm((out_Q[0, :]-amp_lev_mat)**2/2/rx["noise_var"][0]
                                          +  tx["nu_sc"]* amp_lev_mat_sq)
        
        q_est[1,:n,:]   = self.sm((out_I[1,:]-amp_lev_mat)**2/2/rx["noise_var"][1]
                                          +  tx["nu_sc"]* amp_lev_mat_sq)
        
        q_est[1,n:,:]   = self.sm((out_Q[1,:]-amp_lev_mat)**2/2/rx["noise_var"][1]
                                          +  tx["nu_sc"] * amp_lev_mat_sq)

        return q_est, out



            # ================================================ #
            # ================================================ #
            # ================================================ #
            
def train_self(batchNo,tx,fibre,rx,flags):
    
    # minibatch.roll(1,-1)
    # use of the function forward in class twoXtwoFIR, which returns 
    # q_est (the probabilities after soft demapper) and
    # out (the estimated x_hat before soft demapping)
    
    
    # rx[minibatch_real][0][0] = minibatch(HI)
    # rx[minibatch_real][0][1] = minibatch(HQ)
    # rx[minibatch_real][1][0] = minibatch(VI)
    # rx[minibatch_real][1][1] = minibatch(VQ)
    
    
    rx["minibatch_real"]    = rx["sig_real"][:,:,batchNo*rx["NsampBatch"]:(batchNo+1)*rx["NsampBatch"]]
    minibatch_output,out_zf = rx['net'](tx,rx)
    
    # out_train = q
    rx['out_train'][:,:,batchNo*rx["BatchLen"]:(batchNo+1)*rx["BatchLen"]] = minibatch_output.detach().clone()
    
    # out_const = x_hat
    rx['out_const'][:,:,batchNo*rx["BatchLen"]:(batchNo+1)*rx["BatchLen"]] = out_zf.detach().clone()
    
    
    rx['minibatch_out'] = minibatch_output
    rx['batchNo']       = batchNo
    
    
    if flags['plot_const_batch'] == True:
        if rx["Frame"] >=rx["FrameRndRot"]:
            if batchNo%25 == 0:
                # theta_tmp   = gen.rad2deg(fibre['thetas'][rx['Frame']][1],0)
                    
                title       = 'rx - batch {} - frame {}'.format(batchNo,rx['Frame'])
                gen.plot_const_2pol(rx['out_const'],title)
        
    return rx


            # ================================================ #
            # ================================================ #
            # ================================================ #
            
def train_ext(batch,rx,net,tx,out_train,out_const,data):
    
    # minibatch.roll(1,-1)
    # use of the function forward in class twoXtwoFIR, which returns 
    # q_est (the probabilities after soft demapper) and
    # out (the estimated x_hat before soft demapping)
    
    
    # rx[minibatch_real][0][0] = minibatch(HI)
    # rx[minibatch_real][0][1] = minibatch(HQ)
    # rx[minibatch_real][1][0] = minibatch(VI)
    # rx[minibatch_real][1][1] = minibatch(VQ)
    
    
    rx["sig_real"]          = data["rx_real"]
    rx["minibatch_real"]    = rx["sig_real"][:,:,batch*rx["NsampBatch"]:(batch+1)*rx["NsampBatch"]]
    minibatch_output,out_zf = net(tx,rx)
    
    # out_train = q
    out_train[:,:,batch*rx["BatchLen"]:(batch+1)*rx["BatchLen"]] = minibatch_output.detach().clone()
    
    # out_const = x_hat
    out_const[:,:,batch*rx["BatchLen"]:(batch+1)*rx["BatchLen"]] = out_zf.detach().clone()
    
    out = [minibatch_output,out_train,out_const]
    
    return out

            # ================================================ #
            # ================================================ #
            # ================================================ #

def init_dict():
    
    tx          = dict()
    fibre       = dict()
    rx          = dict()
    

    for field_name in ["mod","Nsps","Rs","nu","Ntaps"]:
        tx[field_name] = 0
        
    for field_name in ["channel",'TauPMD','TauCD','phiIQ','theta1','theta_std']:
        fibre[field_name] = 0
        
    for field_name in ["SNRdB","BatchLen","N_lrhalf","Nframes","FrameRndRot","Nmax_SymbFrame"]:
        rx[field_name] = 0

    tx      = misc.sort_dict_by_keys(tx)
    fibre   = misc.sort_dict_by_keys(fibre)
    rx      = misc.sort_dict_by_keys(rx)
    
    return tx,fibre,rx












#%%
# ============================================================================================ #
# ============================================================================================ #
def SER_estimation(tx,rx,out_train,frame):

    # find correlation within 21 symbols
    shift,r                 = kit.find_shift(out_train,tx["symb_real"], 21, tx["amps"], tx["Npolars"])
    
    # compensate Npolars. shift
    out_train               = out_train.roll(r,0)
    
    # compensate time shift (in multiple symb.)
    out_train[0,:,:]        = out_train[0,:,:].roll(int(-shift[0]),-1)
    out_train[1,:,:]        = out_train[1,:,:].roll(int(-shift[1]),-1)
    
    temp_out_train_reshape  = out_train.reshape(tx["Npolars"],2*tx["N_amps"],rx["NBatchFrame"],rx['BatchLen'])
    
    # cut off edge symbols to avoid edge effects
    temp_out_train_cut      = temp_out_train_reshape[:,:,:,:rx['BatchLen']-shift[0]-rx["N_cut"]]
    temp_out_train          = temp_out_train_cut.reshape(tx["Npolars"],2*tx["N_amps"],-1)

    temp_data_tensor        = (tx["symb_real"].reshape(tx["Npolars"],2,rx["NBatchFrame"],rx['BatchLen'])[:,:,:,:rx['BatchLen']-shift[0]-rx["N_cut"]]).reshape(tx["Npolars"],2,-1)

    
    rx["SER_valid"][2:,frame]     = kit.SER_IQflip(temp_out_train[:,:,11:-11-torch.max(torch.abs(shift))],
                                              temp_data_tensor[:,:,11:-11-torch.max(torch.abs(shift))])

    rx['shift'] = shift
    rx['r']     = r
    
    return rx


#%%
# ============================================================================================ #
# ============================================================================================ #

def SNR_estimation(tx,rx):

    rx["SNRdB_est"][rx['Frame']]      = tx["pow_mean"]/torch.mean(rx['Pnoise_batches'])
    rx['Pnoise_est'][:,rx['Frame']]   = torch.mean(rx['Pnoise_batches'],dim=1)  

    return rx




















































# ============================================================================================ #
# ============================================================================================ #
def CPE(y):
    # carrier phase estimation based on Viterbi-Viterbi algorithm
    pi      = torch.tensor(3.141592653589793)
    pi2     = pi/2
    pi4     = pi/4
    M_ma    = 101     # length of moving average filter 
    y_corr  = torch.zeros_like(y)
    y_pow4  = torch.zeros_like(y)
    
    ax      = y[0,0,:]
    bx      = y[0,1,:]
    ay      = y[1,0,:] 
    by      = y[1,1,:]
    
    # taking the signal to the 4th power to cancel out modulation
    # # (a+jb)^4 = a^4 - 6a^2b^2 + b^4 +j(4a^3b - 4ab^3)
    ax2     = ax**2
    bx2     = bx**2
    ay2     = ay**2
    by2     = by**2
    
    y_pow4[0,0,:] = ax2*ax2 - torch.full_like(ax,6)*ax2*bx2 + bx2*bx2 
    y_pow4[0,1,:] = torch.full_like(ax,4)*(ax2*ax*bx - ax*bx2*bx)
    y_pow4[1,0,:] = ay2*ay2 - torch.full_like(ay,6)*ay2*by2 + by2*by2 
    y_pow4[1,1,:] = torch.full_like(ay,4)*(ay2*ay*by - ay*by2*by)
    
    # moving average filtering
    kernel_ma     = torch.full((1,1,M_ma), 1/M_ma, device=y.device, dtype=torch.float32)
    y_conv        = misc.my_zeros_tensor((4,1,y_pow4.shape[2]))
    
    y_conv[0,0,:] = y_pow4[0,0,:]
    y_conv[1,0,:] = y_pow4[0,1,:]
    y_conv[2,0,:] = y_pow4[1,0,:]
    y_conv[3,0,:] = y_pow4[1,1,:]

    y_ma        = F.conv1d(y_conv,kernel_ma,bias=None,padding=M_ma//2)

    phiX_corr   = torch.atan2(y_ma[1,0,:],-y_ma[0,0,:])/4
    diff_phiX   = phiX_corr[1:] - phiX_corr[:-1]

    ind_X_pos   = torch.nonzero(diff_phiX>pi4)
    ind_X_neg   = torch.nonzero(diff_phiX<-pi4)

    for i in ind_X_pos:     # unwrapping
        phiX_corr[i+1:] -=  pi2
    for j in ind_X_neg:
        phiX_corr[j+1:] +=  pi2

    cos_phiX    = torch.cos(phiX_corr)
    sin_phiX    = torch.sin(phiX_corr)

    phiY_corr   = torch.atan2(y_ma[3,0,:],-y_ma[2,0,:])/4
    diff_phiY   = phiY_corr[1:] - phiY_corr[:-1]

    ind_Y_pos   = torch.nonzero(diff_phiY>pi4)
    ind_Y_neg   = torch.nonzero(diff_phiY<-pi4)
    
    for ii in ind_Y_pos:    # unwrapping 
        phiY_corr[ii+1:] -=  pi2
    for jj in ind_Y_neg:
        phiY_corr[jj+1:] +=  pi2

    cos_phiY = torch.cos(phiY_corr)
    sin_phiY = torch.sin(phiY_corr)

    # compensating phase offset
    y_corr[0,0,:] = ax*cos_phiX - bx*sin_phiX
    y_corr[0,1,:] = bx*cos_phiX + ax*sin_phiX
    y_corr[1,0,:] = ay*cos_phiY - by*sin_phiY
    y_corr[1,1,:] = by*cos_phiY + ay*sin_phiY
    
    return y_corr #, phiX_corr, phiY_corr

# ============================================================================================ #
# estimate symbol error rate from estimated a posterioris q
# ============================================================================================ #

def SER_IQflip(q, tx): 
    device = q.device
    N_amp = q.shape[1]//2
    dec = torch.empty_like(tx, device=device, dtype=torch.int16)
    data = torch.empty_like(tx, device=device, dtype=torch.int16)
    data_IQinv = torch.empty_like(data)
    SER = torch.ones(2,2,4, device=device, dtype=torch.float32)
    
    scale = (N_amp-1)/2
    data = torch.round(scale*tx.float()+scale) # decode TX
    data_IQinv[:,0,:], data_IQinv[:,1,:] = data[:,0,:], -(data[:,1,:]-scale*2)  # compensate potential IQ flip
    ### zero phase-shift
    dec[:,0,:], dec[:,1,:] = torch.argmax(q[:,:N_amp,:], dim=1), torch.argmax(q[:,N_amp:,:], dim=1) # hard decision on max(q)
    SER[0,:,0] = torch.mean( ((data - dec).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    SER[1,:,0] = torch.mean( ((data_IQinv - dec).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    
    ### pi phase-shift
    dec_pi = -(dec-scale*2)
    SER[0,:,1] = torch.mean( ((data - dec_pi).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    SER[1,:,1] = torch.mean( ((data_IQinv - dec_pi).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)

    ### pi/4 phase-shift
    dec_pi4 = torch.empty_like(dec)
    dec_pi4[:,0,:], dec_pi4[:,1,:] = -(dec[:,1,:]-scale*2), dec[:,0,:]
    SER[0,:,2] = torch.mean( ((data - dec_pi4).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    SER[1,:,2] = torch.mean( ((data_IQinv - dec_pi4).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)

    ### 3pi/4 phase-shift
    dec_3pi4 = -(dec_pi4-scale*2)
    SER[0,:,3] = torch.mean( ((data - dec_3pi4).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)
    SER[1,:,3] = torch.mean( ((data_IQinv - dec_3pi4).bool().any(dim=1)).to(torch.float), dim=-1, dtype=torch.float32)

    SER_out = torch.amin(SER, dim=(0,-1))   # choose minimum estimation per Npolarsarisation
    return SER_out 

# ============================================================================================ #
# estimate symbol error rate from output constellation by considering PCS
# ============================================================================================ #

def SER_constell_shaping(rx, tx, amp_levels, nu_sc, var): 

    device = rx.device
    N_amp = amp_levels.shape[0]
    data = torch.empty_like(tx, device=device, dtype=torch.int32)
    data_IQinv = torch.empty_like(data)
    SER = torch.ones(2,2,4, device=device, dtype=torch.float32)

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

# ============================================================================================ #
# hard decision based on the decision boundaries d_vec0 (lower) and d_vec1 (upper)
# ============================================================================================ #

    
def dec_on_bound(rx,tx_int,d_vec0, d_vec1):
    SER = torch.zeros(rx.shape[0], dtype = torch.float32, device = rx.device)
    
    xI0 = d_vec0.index_select(dim=0,index=tx_int[0,0,:])
    xI1 = d_vec1.index_select(dim=0,index=tx_int[0,0,:])
    corr_xI = torch.bitwise_and((xI0 <= rx[0, 0, :]), (rx[0, 0, :] < xI1))
    xQ0 = d_vec0.index_select(dim=0,index=tx_int[0,1,:])
    xQ1 = d_vec1.index_select(dim=0,index=tx_int[0,1,:])
    corr_xQ = torch.bitwise_and((xQ0 <= rx[0, 1, :]), (rx[0, 1, :] < xQ1))

    yI0 = d_vec0.index_select(dim=0,index=tx_int[1,0,:])
    yI1 = d_vec1.index_select(dim=0,index=tx_int[1,0,:])
    corr_yI = torch.bitwise_and((yI0 <= rx[1, 0, :]), (rx[1, 0, :] < yI1))
    yQ0 = d_vec0.index_select(dim=0,index=tx_int[1,1,:])
    yQ1 = d_vec1.index_select(dim=0,index=tx_int[1,1,:])
    corr_yQ = torch.bitwise_and((yQ0 <= rx[1, 1, :]), (rx[1, 1, :] < yQ1))

    ex, ey = ~(torch.bitwise_and(corr_xI,corr_xQ)), ~(torch.bitwise_and(corr_yI,corr_yQ))   # no error only if both I or Q are correct
    SER[0], SER[1] = torch.sum(ex)/ex.nelement(), torch.sum(ey)/ey.nelement()   # SER = numb. of errors/ num of symbols
    return SER

# ============================================================================================ #
# find shiftings in both Npolarsarisation and time by correlation with expectation of x^I with respect to q
# ============================================================================================ #

def find_shift(q, tx, N_shift, amp_levels, Npolars): 
    corr_max = torch.empty(2,2,2, device = q.device, dtype=torch.float32)
    N_amp = q.shape[1]//2
    corr_ind = torch.empty_like(corr_max)
    len_corr = q.shape[-1] 
    amTT = amp_levels.repeat(Npolars,len_corr,1).transpose(1,2)
    E = torch.sum(amTT * q[:,:N_amp,:len_corr], dim=1) # calculate expectation E_q(x^I) of in-phase component

    # correlate with (both Npolarsarisations) and shifted versions in time --> find max. correlation
    E_mat = torch.empty(2,len_corr,N_shift, device=q.device, dtype=torch.float32)
    for i in range(N_shift):
        E_mat[:,:,i] = torch.roll(E,i-N_shift//2,-1)
    corr_max[0,:,:], corr_ind[0,:,:] = torch.max( torch.abs(tx[:,0,:len_corr].float() @ E_mat), dim=-1)
    corr_max[1,:,:], corr_ind[1,:,:] = torch.max( torch.abs(tx[:,1,:len_corr].float() @ E_mat), dim=-1) 
    corr_max, ind_max = torch.max(corr_max,dim=0); #corr_ind = corr_ind[ind_max]

    ind_XY = torch.zeros(2,device=q.device, dtype=torch.int16); ind_YX = torch.zeros_like(ind_XY)
    ind_XY[0] = corr_ind[ind_max[0,0],0,0]; ind_XY[1] = corr_ind[ind_max[1,1],1,1]
    ind_YX[0] = corr_ind[ind_max[0,1],0,1]; ind_YX[1] = corr_ind[ind_max[1,0],1,0] 

    if (corr_max[0,0]+corr_max[1,1]) >= (corr_max[0,1]+corr_max[1,0]):
        return N_shift//2-ind_XY, 0
    else:
        return N_shift//2-ind_YX, 1

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

# ============================================================================================ #
# constant modulus algorithm
# ============================================================================================ #



def CMA(Rx, R, h, lr, sps, eval):
    device  = Rx.device
    M       = h.shape[-1]
    N       = Rx.shape[-1]
    mh      = M//2       

    # zero-padding
    pad     = torch.zeros(2,2,mh, device = device, dtype=torch.float32)
    y       = torch.cat((pad, Rx, pad), -1)
    y       /= torch.mean(y[:,0,:]**2 + y[:,1,:]**2 )     # scaling for mapping to R=1

    out     = torch.zeros(2,2,N//sps, device = device, dtype=torch.float32)
    e       = torch.empty(N//sps, 2, device = device, dtype=torch.float32)

    for i in torch.arange(mh,N+mh,sps):     # downsampling included
        ind = torch.arange(-mh+i,i+mh+1)
        k   = i//sps - mh

        # 2x2 butterfly FIR
        # Estimate Symbol 
        out[0,0,k] = (
            torch.matmul(y[0,0,ind],h[0,0,0,:]) - torch.matmul(y[0,1,ind],h[0,0,1,:]) +
            torch.matmul(y[1,0,ind],h[0,1,0,:]) - torch.matmul(y[1,1,ind],h[0,1,1,:])  
        )
        out[1,0,k] = (
            torch.matmul(y[0,0,ind],h[1,0,0,:]) - torch.matmul(y[0,1,ind],h[1,0,1,:]) + 
            torch.matmul(y[1,0,ind],h[1,1,0,:]) - torch.matmul(y[1,1,ind],h[1,1,1,:]) 
        )
        # Estimate Symbol 
        out[0,1,k] = (
            torch.matmul(y[0,0,ind],h[0,0,1,:]) + torch.matmul(y[0,1,ind],h[0,0,0,:]) + 
            torch.matmul(y[1,0,ind],h[0,1,1,:]) + torch.matmul(y[1,1,ind],h[0,1,0,:])  
        )
        out[1,1,k] = (
            torch.matmul(y[0,0,ind],h[1,0,1,:]) + torch.matmul(y[0,1,ind],h[1,0,0,:]) + 
            torch.matmul(y[1,0,ind],h[1,1,1,:]) + torch.matmul(y[1,1,ind],h[1,1,0,:]) 
        )
    
        # Calculate error        
        e[k,0] = R - out[0,0,k]**2 - out[0,1,k]**2
        e[k,1] = R - out[1,0,k]**2 - out[1,1,k]**2
        
        if eval == True:
        # Update filters
        #for i in range(2):
            h[0,0,0,:] += 2*lr*e[k,0]* ( out[0,0,k]*y[0,0,ind] + out[0,1,k]*y[0,1,ind])       
            h[0,0,1,:] += 2*lr*e[k,0]* ( out[0,1,k]*y[0,0,ind] - out[0,0,k]*y[0,1,ind])
            h[0,1,0,:] += 2*lr*e[k,0]* ( out[0,0,k]*y[1,0,ind] + out[0,1,k]*y[1,1,ind])
            h[0,1,1,:] += 2*lr*e[k,0]* ( out[0,1,k]*y[1,0,ind] - out[0,0,k]*y[1,1,ind])

            h[1,0,0,:] += 2*lr*e[k,1]* ( out[1,0,k]*y[0,0,ind] + out[1,1,k]*y[0,1,ind]) 
            h[1,0,1,:] += 2*lr*e[k,1]* ( out[1,1,k]*y[0,0,ind] - out[1,0,k]*y[0,1,ind])
            h[1,1,0,:] += 2*lr*e[k,1]* ( out[1,0,k]*y[1,0,ind] + out[1,1,k]*y[1,1,ind])
            h[1,1,1,:] += 2*lr*e[k,1]* ( out[1,1,k]*y[1,0,ind] - out[1,0,k]*y[1,1,ind])
    return out, h, e

# ============================================================================================ #
# ============================================================================================ #

def CMAbatch(Rx, R, h, lr, batchlen, sps, eval):
    device = Rx.device
    M = h.shape[-1]
    N = Rx.shape[-1]
    mh = M//2       

    # zero-padding
    pad = torch.zeros(2,2,mh, device = device, dtype=torch.float32)
    y = torch.cat((pad, Rx, pad), -1)
    y /= torch.mean(y[:,0,:]**2 + y[:,1,:]**2 )

    out = torch.zeros(2,2,N//sps, device = device, dtype=torch.float32)
    e = torch.empty(N//sps, 2, device = device, dtype=torch.float32)

    buf = torch.empty(2,2,2,N//sps, M, device = device, dtype=torch.float32)


    for i in torch.arange(mh,N+mh,sps):     # downsampling included
        ind = torch.arange(-mh+i,i+mh+1)
        k = i//sps - mh

        # 2x2 butterfly FIR
        out[0,0,k] = torch.matmul(y[0,0,ind],h[0,0,0,:]) - torch.matmul(y[0,1,ind],h[0,0,1,:]) + torch.matmul(y[1,0,ind],h[0,1,0,:]) - torch.matmul(y[1,1,ind],h[0,1,1,:])  # Estimate Symbol 
        out[1,0,k] = torch.matmul(y[0,0,ind],h[1,0,0,:]) - torch.matmul(y[0,1,ind],h[1,0,1,:]) + torch.matmul(y[1,0,ind],h[1,1,0,:]) - torch.matmul(y[1,1,ind],h[1,1,1,:]) 
        
        out[0,1,k] = torch.matmul(y[0,0,ind],h[0,0,1,:]) + torch.matmul(y[0,1,ind],h[0,0,0,:]) + torch.matmul(y[1,0,ind],h[0,1,1,:]) + torch.matmul(y[1,1,ind],h[0,1,0,:])  # Estimate Symbol 
        out[1,1,k] = torch.matmul(y[0,0,ind],h[1,0,1,:]) + torch.matmul(y[0,1,ind],h[1,0,0,:]) + torch.matmul(y[1,0,ind],h[1,1,1,:]) + torch.matmul(y[1,1,ind],h[1,1,0,:]) 

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

            if (k%batchlen == 0 and k!=0):  # batch-wise updating
                h[0,0,0,:] += 2*lr*torch.sum(torch.mul(buf[0,0,0,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1)       # Update filters
                h[0,0,1,:] += 2*lr*torch.sum(torch.mul(buf[0,0,1,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 
                h[0,1,0,:] += 2*lr*torch.sum(torch.mul(buf[0,1,0,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 
                h[0,1,1,:] += 2*lr*torch.sum(torch.mul(buf[0,1,1,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 

                h[1,0,0,:] += 2*lr*torch.sum(torch.mul(buf[1,0,0,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,0,1,:] += 2*lr*torch.sum(torch.mul(buf[1,0,1,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,1,0,:] += 2*lr*torch.sum(torch.mul(buf[1,1,0,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,1,1,:] += 2*lr*torch.sum(torch.mul(buf[1,1,1,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
    return out, h, e

# ============================================================================================ #
# ============================================================================================ #

def CMAflex(Rx, R, h, lr, batchlen, symb_step, sps, eval):
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
            if (k%symb_step == 0 and k>=batchlen):
                
                # Update filters
                h[0,0,0,:] += 2*lr*torch.sum(torch.mul(buf[0,0,0,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1)
                h[0,0,1,:] += 2*lr*torch.sum(torch.mul(buf[0,0,1,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 
                h[0,1,0,:] += 2*lr*torch.sum(torch.mul(buf[0,1,0,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 
                h[0,1,1,:] += 2*lr*torch.sum(torch.mul(buf[0,1,1,k-batchlen:k,:].T,e[k-batchlen:k,0]),dim=1) 

                h[1,0,0,:] += 2*lr*torch.sum(torch.mul(buf[1,0,0,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,0,1,:] += 2*lr*torch.sum(torch.mul(buf[1,0,1,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,1,0,:] += 2*lr*torch.sum(torch.mul(buf[1,1,0,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
                h[1,1,1,:] += 2*lr*torch.sum(torch.mul(buf[1,1,1,k-batchlen:k,:].T,e[k-batchlen:k,1]),dim=1) 
    return out, h, e


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


    

# convolution
# x = [1 2 3]
# y = [4 5 6]
# z = x*y =  [z1 z2 z3 z4 z5]

# what is done:
#    z = fliplr(z): [6 5 4]
# 
# z1 = 
# [0 0 1 2 3].
# [6 5 4 0 0] = 6.0+5.0+4.1+2.0+3.0 = 4
#
# z2 = 
# [0 1 2 3].
# [6 5 4 0] = 6.0+5.1+4.2+3.0       = 13
#
# z3 = 
# [1 2 3].
# [6 5 4] = 6.1+5.2+4.3             = 28
#
# z4 = 
# [1 2 3 0].
# [0 6 5 4] = 0.1+6.2+5.3+4.0       = 27
#
# z5 = 
# [1 2 3 0 0].
# [0 0 6 5 4] = 0.1+0.2+6.3+5.0+4.0 = 18
#
# mode = full (default) [4 13 28 27 18]
# mode = same           [13 28 27]
# mode = valid          [28]









