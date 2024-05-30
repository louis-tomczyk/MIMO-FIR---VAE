# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          : 2024-03-04 (1.0.0)    creation
#                   : 2024-04-01 (1.1.1)    [NEW] plot_fir
#                   : 2024-04-03 (1.2.0)    plot_const_2pol
#                                           [NEW] show_fir_central_tap
#                   : 2024-04-07 (1.2.1)    plot_const_1d, plot_const_2d
#                   : 2024-05-21 (1.3.0)    [NEW] plot_decisions, plot_xcorr_2x2
#                   : 2024-05-21 (1.3.1)    plot_loss_cma --- y = -y if mean(y[0:10])<0
#                   : 2024-05-24 (1.4.0)    plot_const_1pol --- cplx to real processing
#                                           [NEW] plot_const_2pol_2sig
#   Date            : 2024-05-24 (1.4.1)    plot_const_2pol_2sig, plot_const_1/2pol (Npoints max)
#   Version         : 1.4.1
#   Licence         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute - place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source - include copyright - state changes - include license
#
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

#%% =============================================================================
# --- CONTENTS ---
# - fir_2Dto3D
# - fir_3Dto2D
# - plot_const_1pol
# - plot_const_2pol
# - plot_const_2pol_2sig    (1.4.0)
# - plot_decisions          (1.3.0)
# - plot_fir                (1.1.1)
# - plot_loss_batch
# - plot_loss_cma
# - plot_losses
# - plot_phases
# - plot_xcorr_2x2          (1.3.0)
# - real2complex_fir
# - show_fir_central_tap    (1.2.0)
# =============================================================================

#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import torch
import lib_misc as misc
import lib_matlab as mb

pi                              = np.pi
fig_width                       = 10
fig_height                      = fig_width/1.618
fig_resolution                  = fig_width*fig_height


plt.rcParams['figure.figsize']  = (fig_width, fig_height*2)
plt.rcParams['figure.dpi']      = fig_resolution*5
plt.rcParams['font.weight']     = "normal"
plt.rcParams['axes.labelweight']= "bold"
plt.rcParams['axes.linewidth']  = "0.1"
plt.rcParams["axes.titlesize"]  = "20"
plt.rcParams["axes.labelsize"]  = "12"
plt.rcParams["axes.titleweight"]= "bold"

#%% ===========================================================================
# --- FUNCTIONS
# =============================================================================

#%% 
def fir_2Dto3D(rx):
    
    if type(rx) == dict:
        fir = rx['h_est']
    else:
        fir = rx
    
    # if len(fir.shape)
    Ntaps       = len(fir.transpose())
    tmp         = np.zeros((2,2,Ntaps),dtype = complex)
    
    tmp[0,0,:]    = fir[0,:] # HH
    tmp[0,1,:]    = fir[1,:] # VH
    tmp[1,0,:]    = fir[2,:] # VH
    tmp[1,1,:]    = fir[3,:] # VV
    
    return tmp

#%% 
def fir_3Dto2D(rx):
    
    if type(rx) == dict:
        fir = rx['h_est']
    else:
        fir = rx
        
    if type(fir) == torch.Tensor:
        fir         = fir.detach().numpy()
    
    fir_shape   = fir.shape
    Ntaps       = fir_shape[-1]
    tmp         = np.zeros(4,Ntaps)
    
    tmp[0,:]    = fir[0,0,:]
    tmp[1,:]    = fir[0,1,:]
    tmp[2,:]    = fir[1,0,:]
    tmp[3,:]    = fir[1,1,:]
    
    return tmp



#%%

def plot_const_1pol(sig,norm = 0,*varargin):

    if type(sig) == torch.Tensor:
        sig = sig.detach().numpy()
        
    # should be kept if SIG is not at 1 sample per symbol
    Nsps    = 2
    Nmax    = int(5e3)
    
    if np.isrealobj(sig) == 0:
        polHI   = np.real(sig[0::Nsps])
        polHQ   = np.imag(sig[0::Nsps])

    else:
        polHI   = sig[0]
        polHQ   = sig[1]
        polVI   = sig[2]
        polVQ   = sig[3]

    if norm == 1:
        M1HI         = np.mean(abs(polHI)**2)
        M1HQ         = np.mean(abs(polHQ)**2)
    
        MMM         = np.sqrt(max([M1HI,M1HQ]))
    else:
        MMM         = 1

    polHInorm   = polHI/MMM
    polHQnorm   = polHQ/MMM

    plt.figure()

    plt.scatter(polHInorm[0::Nsps], polHQnorm[0::Nsps],c='black')
    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)
    plt.xlabel("in phase")
    plt.ylabel("quadrature")
    
    if len(varargin) == 1:
        plt.title('polH ' + varargin[0])
        
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
        
    
#%%
def plot_const_2pol(sig,norm = 0,*varargin):
    
    if type(sig) == torch.Tensor:
        sig = sig.detach().numpy()
        
        
    # should be kept if SIG is not at 1 sample per symbol
    Nsps    = 2
    Nmax    = int(5e3)
    
    if np.isrealobj(sig) == 0:
        polHI   = np.real(sig[0])
        polHQ   = np.imag(sig[0])
        polVI   = np.real(sig[1])
        polVQ   = np.imag(sig[1])

    else:
        polHI   = sig[0]
        polHQ   = sig[1]
        polVI   = sig[2]
        polVQ   = sig[3]
        
    if norm == 1:
        M1HI         = np.mean(abs(polHI)**2)
        M1HQ         = np.mean(abs(polHQ)**2)
        M1VI         = np.mean(abs(polVI)**2)
        M1VQ         = np.mean(abs(polVQ)**2)
    
        MMM         = np.sqrt(max([M1HI,M1HQ,M1VI,M1VQ]))
    else:
        MMM         = 1
        
    if mb.numel(polHI)>Nmax:
        polHI = polHI[0:Nmax]
        polHQ = polHQ[0:Nmax]
        polVI = polVI[0:Nmax]
        polVQ = polVQ[0:Nmax]

    polHInorm   = polHI/MMM
    polHQnorm   = polHQ/MMM
    polVInorm   = polVI/MMM
    polVQnorm   = polVQ/MMM

    plt.figure()
    
    ### ------- SUBPLOT 1
    plt.subplot(1,2,1)
    plt.scatter(polHInorm[0::Nsps], polHQnorm[0::Nsps],c='black')
    
    # if len(varargin) == 2:
    #     if varargin[1]['mod'].lower() == "64qam":
    #         plt.xlim(-2.5,2.5)
    #         plt.ylim(-2.5,2.5)
    #     else:
    #         plt.xlim(-2.5, 2.5)
    #         plt.ylim(-2.5, 2.5)
    # else:
    #     plt.xlim(-2.5, 2.5)
    #     plt.ylim(-2.5, 2.5)
        
    plt.xlabel("in phase")
    plt.ylabel("quadrature")
    
    if len(varargin) !=0:
        plt.title('polH ' + varargin[0])
    plt.gca().set_aspect('equal', adjustable='box')
        
    ### ------- SUBPLOT 2
    plt.subplot(1,2,2)
    plt.scatter(polVInorm[0::Nsps], polVQnorm[0::Nsps],c='black')
    
    # if len(varargin) == 2:
    #     if varargin[1]['mod'].lower() == "64qam":
    #         plt.xlim(-2.5,2.5)
    #         plt.ylim(-2.5,2.5)
    #     else:
    #         plt.xlim(-2.5, 2.5)
    #         plt.ylim(-2.5, 2.5)
    # else:
    #     plt.xlim(-2.5, 2.5)
    #     plt.ylim(-2.5, 2.5)
        
    plt.xlabel("in phase")
    
    if len(varargin) != 0:
        plt.title('polV '+varargin[0])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()
    
#%%
def plot_const_2pol_2sig(sig1,sig2,labels,norm = 0,*varargin):

    if type(sig1) == torch.Tensor:
        sig1 = sig1.detach().numpy()

    if type(sig2) == torch.Tensor:
        sig2 = sig2.detach().numpy()

    
    # should be kept if SIG is not at 1 sample per symbol
    Nsps    = 2
    Nmax    = int(5e3)
    
    if np.isrealobj(sig1) == 0:
        pol1HI   = np.real(sig1[0]).flatten()
        pol1HQ   = np.imag(sig1[0]).flatten()
        pol1VI   = np.real(sig1[1]).flatten()
        pol1VQ   = np.imag(sig1[1]).flatten()

    else:
        pol1HI   = sig1[0].flatten()
        pol1HQ   = sig1[1].flatten()
        pol1VI   = sig1[2].flatten()
        pol1VQ   = sig1[3].flatten()
        
    if np.isrealobj(sig2) == 0:
        pol2HI   = np.real(sig2[0]).flatten()
        pol2HQ   = np.imag(sig2[0]).flatten()
        pol2VI   = np.real(sig2[1]).flatten()
        pol2VQ   = np.imag(sig2[1]).flatten()

    else:
        pol2HI   = sig2[0].flatten()
        pol2HQ   = sig2[1].flatten()
        pol2VI   = sig2[2].flatten()
        pol2VQ   = sig2[3].flatten()
        
    if mb.numel(pol1HI)>Nmax:
        pol1HI = pol1HI[0:Nmax]
        pol1HQ = pol1HQ[0:Nmax]
        pol1VI = pol1VI[0:Nmax]
        pol1VQ = pol1VQ[0:Nmax]

        pol2HI = pol2HI[0:Nmax]
        pol2HQ = pol2HQ[0:Nmax]
        pol2VI = pol2VI[0:Nmax]
        pol2VQ = pol2VQ[0:Nmax]

    if norm == 1:
        M1HI         = np.mean(abs(pol1HI)**2)
        M1HQ         = np.mean(abs(pol1HQ)**2)
        M1VI         = np.mean(abs(pol1VI)**2)
        M1VQ         = np.mean(abs(pol1VQ)**2)
    
        MMM         = np.sqrt(max([M1HI,M1HQ,M1VI,M1VQ]))
    else:
        MMM         = 1

    pol1HInorm   = pol1HI/MMM
    pol1HQnorm   = pol1HQ/MMM
    pol1VInorm   = pol1VI/MMM
    pol1VQnorm   = pol1VQ/MMM

    pol2HInorm   = pol2HI/MMM
    pol2HQnorm   = pol2HQ/MMM
    pol2VInorm   = pol2VI/MMM
    pol2VQnorm   = pol2VQ/MMM

    plt.figure()

    ### ------- SUBPLOT 1
    plt.subplot(1,2,1)
    plt.scatter(pol1HInorm[0::Nsps], pol1HQnorm[0::Nsps],c='black',label=labels[0])
    plt.scatter(pol2HInorm[0::Nsps], pol2HQnorm[0::Nsps],c='blue',label=labels[1])

    plt.xlabel("in phase")
    plt.ylabel("quadrature")

    if len(varargin) !=0:
        plt.title('polH ' + varargin[0])
    plt.gca().set_aspect('equal', adjustable='box')

    ### ------- SUBPLOT 2
    plt.subplot(1,2,2)
    plt.scatter(pol1VInorm[0::Nsps], pol1VQnorm[0::Nsps],c='black',label=labels[0])
    plt.scatter(pol2VInorm[0::Nsps], pol2VQnorm[0::Nsps],c='blue',label=labels[1])


    plt.xlabel("in phase")

    if len(varargin) != 0:
        plt.title('polV '+varargin[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    plt.show()

#%%
def plot_decisions(t,r,Nplots):
    
    for k in range(Nplots):
        a = k*100
        b = a+25
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(t[0][a:b])
        plt.plot(r[0][a:b],linestyle="dashed")
        plt.title('HI = a = {} --- b = {}'.format(a,b))

        plt.subplot(2,2,2)
        plt.plot(t[1][a:b])
        plt.plot(r[1][a:b],linestyle="dashed")
        plt.title('HQ = a = {} --- b = {}'.format(a,b))

        plt.subplot(2,2,3)
        plt.plot(t[2][a:b])
        plt.plot(r[2][a:b],linestyle="dashed")
        plt.title('VI = a = {} --- b = {}'.format(a,b))

        plt.subplot(2,2,4)
        plt.plot(t[3][a:b])
        plt.plot(r[3][a:b],linestyle="dashed")
        plt.title('VQ = a = {} --- b = {}'.format(a,b))
        
        plt.show()


#%%
def plot_fir(rx,*varargin):

    if type(rx) == dict:
        Ntaps = max(rx['h_est'].shape)

        if len(rx["h_est"].shape) == 4:
            rx      = real2complex_fir(rx)
            myfir   = abs(rx['h_est_cplx'])

        else:
            myfir   = abs(rx["h_est"])

    else:
        myfir2 = np.abs(rx)
        if len(rx.shape) == 1:
            Ntaps       = len(rx.shape)
            myfir       = np.zeros((4,Ntaps))
            myfir[0,:]  = myfir2
            myfir[3,:]  = myfir2            
        else:
            Ntaps       = max(myfir2.shape)
            myfir       = myfir2


    Taps = np.linspace(1,Ntaps,Ntaps)-round(Ntaps/2)-1
    plt.figure()
    
    
    plt.subplot(1,2,1)
    plt.plot(Taps,myfir[0],label='h_{11}',linestyle = "solid",linewidth = 1.5,color = 'black')
    plt.plot(Taps,myfir[3],label='h_{22}',linestyle = "dashed",linewidth = 1.5,color = 'gray')
    plt.ylim((0,1))
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(Taps,myfir[1],label='h_{12}',linestyle = "solid",linewidth = 1.5,color = 'black')
    plt.plot(Taps,myfir[2],label='h_{21}',linestyle = "dashed",linewidth = 1.5,color = 'gray')
    plt.ylim((0,1))
    plt.legend()
    
    if len(varargin) != 0:
        plt.title(varargin[0])
        
    plt.show()
    
    

#%%

def plot_loss_batch(rx,flags,saving,keyword,what):
    
    # a tensor with GRAD on cannot be plotted, it requires first to put it in a normal array
    # that's the purpose of DETACH
    
    # do not intervert SAVEFIG and SHOW otherwise the figure will not be saved

    if flags['plot_loss_batch']:

        Binary  = misc.string_to_binary(what)
        Decimal = misc.binary_to_decimal(Binary)
        Fig_id  = int(Decimal*1e-12)
        plt.figure(Fig_id)
        
        x = [k for k in range(rx['BatchNo'])]
        y = rx[what+'_subframe'][rx['Frame']][:rx['BatchNo']].detach().numpy()        
        
        if rx['Frame'] >= rx['FrameChannel']:
            linestyle   = "solid"
        else:
            linestyle = ":"
         
        if rx['Frame'] == rx['FrameChannel']:
            plt.plot(x,y,linestyle=linestyle,linewidth = 5, color = 'k',label = 'frame {} - after channel 1st round'.format(rx["Frame"]))   
         
        elif rx['Frame']%saving['skip'] == 0:
            if rx["Frame"]<rx['FrameChannel']:
                plt.plot(x,y,linestyle=linestyle,label = 'frame {} - b4 channel'.format(rx["Frame"]))
            else:
                plt.plot(x,y,linestyle=linestyle,label = 'frame {} - after channel'.format(rx["Frame"]))

        if rx['Frame'] == rx['Nframes']-1:
            plt.xlabel("iteration")
            plt.ylabel(what)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
            
            if keyword != []:
                patterns        = []
                pattern_values  = []
                
                for kw in keyword:
                    pattern_index   = misc.find_string(kw,saving['filename'])
                    patterns.append(saving['filename'][pattern_index[0]:pattern_index[1]])
                    pattern_values.append(saving['filename'][pattern_index[1]+1:pattern_index[1]+1+2])
                
                tname = what+' in batches - '
                for k in range(len(patterns)):
                    tname = tname + "{}-{} -- ".format(patterns[k],pattern_values[k])
                    
                plt.title(tname)

            else:
                plt.title(what+" in the batches")
                
            output_file = "{}.png".format(saving['filename'])
            plt.savefig(output_file,bbox_inches='tight')
            
#%%

def plot_loss_cma(rx,flags,saving,keyword,what):
    
    # a tensor with GRAD on cannot be plotted, it requires first to put it in a normal array
    # that's the purpose of DETACH
    
    # do not intervert SAVEFIG and SHOW otherwise the figure will not be saved

    if what.lower() == "x" or what.lower() == "polX":
        ind = 0
    elif what.lower() == "y" or what.lower() == "polY":
        ind = 1
    else:
        ind = [1,2]
        
    plt.figure(0)

    Nplot = int(rx['NSymbFrame']/5)
    
    x = np.linspace(0, Nplot-1, Nplot)
    y = rx["CMA"]["losses"][str(rx['Frame'])].transpose()
    y = y[ind][:Nplot].transpose()
    
    if np.mean(y[0:10])<0:
        y = -y
    
    if rx['Frame'] >= rx['FrameChannel']:
        linestyle   = "solid"
    else:
        linestyle = ":"
     
    if rx['Frame'] == rx['FrameChannel']:
        plt.plot(x,y,linestyle=linestyle,linewidth = 5, color = 'k',label = 'frame {} - after channel 1st round'.format(rx["Frame"]))   
     
    elif rx['Frame']%saving['skip'] == 0:
        if rx["Frame"]<rx['FrameChannel']:
            plt.plot(x,y,linestyle=linestyle,linewidth = 1,label = 'frame {} - b4 channel'.format(rx["Frame"]))
        else:
            plt.plot(x,y,linestyle=linestyle,linewidth = 1,label = 'frame {} - after channel'.format(rx["Frame"]))

    if rx['Frame'] == rx['Nframes']-1:
        plt.xlabel("N symbols")
        plt.ylabel("Loss")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    
        
        if keyword != []:
            patterns        = []
            pattern_values  = []
            
            for kw in keyword:
                pattern_index   = misc.find_string(kw,saving['filename'])
                patterns.append(saving['filename'][pattern_index[0]:pattern_index[1]])
                pattern_values.append(saving['filename'][pattern_index[1]+1:pattern_index[1]+1+2])
            
            tname = what+' '
            for k in range(len(patterns)):
                tname = tname + "{}-{} -- ".format(patterns[k],pattern_values[k])
                
            plt.title(tname)

        else:
            plt.title(what+' ')
            
        plt.ylim((-2,max(10,max(y))))
        output_file = "{}.png".format(saving['filename']+ ' '+what+'_batch')
        plt.savefig(output_file,bbox_inches='tight')




#%%

def plot_losses(Losses,OSNRdBs,title):
    
        # Création de la grille de sous-graphiques avec une résolution DPI élevée
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=400)
        
        # Tracer la première figure dans le premier sous-graphique
        axes[0].plot(Losses)
        axes[0].set_xlabel('iteration')
        axes[0].set_ylabel('loss')
        axes[0].set_title('_')
        axes[0].set_ylim(-1200, 650)
        
        # Tracer la deuxième figure dans le deuxième sous-graphique
        axes[1].plot(OSNRdBs)
        axes[1].set_xlabel('iteration')
        axes[1].set_ylabel('OSNRdB [dB]')
        axes[1].set_title('_')
        axes[1].set_ylim(0, 30)

        # Ajuster l'espacement entre les sous-graphiques
        plt.tight_layout()
        fig.suptitle(title, y=1.02)        

        plt.savefig(title,dpi=400,format = 'png')
        
        

#%%

def plot_phases(tx,rx):

    tmpTX   = tx["phisBatch"][rx['FrameChannel']:,:]
    tmpRX   = rx["phases_est_mat"][rx['FrameChannel']:,:]
    
    tmpTX   = np.unwrap(np.reshape(tmpTX,(-1,1)).squeeze())*180/pi
    tmpRX   = np.unwrap(np.reshape(tmpRX,(-1,1)).squeeze())*180/pi
    

    plt.plot(tmpTX,label = "TX")
    plt.plot(tmpRX,label = "RX")
    plt.legend()
    plt.title(rx['title phases'])
    plt.show()
    
    

#%%

def plot_xcorr_2x2(sig_11,sig_12,sig_21, sig_22,title,ref,zoom):

    assert sig_11.shape == sig_12.shape == sig_21.shape == sig_22.shape
    
    Nsamp = len(sig_11)
    # displaying the correaltions to check
    x = np.linspace(0,Nsamp,Nsamp)-Nsamp/2
    
    plt.subplot(2,2,1)
    plt.plot(x,sig_11)
    plt.title(title+ ' 11')
    
    if ref == 1:
        plt.plot([0,0],[-250,max(abs(sig_11))],c='black',linestyle = "dotted")
        if zoom == 1:
            plt.xlim(-100,100)

    plt.subplot(2,2,2)
    plt.plot(x,sig_12)
    plt.title(title+ ' 12')
    if ref == 1:
        plt.plot([0,0],[-250,max(abs(sig_12))],c='black',linestyle = "dotted")
        if zoom == 1:
            plt.xlim(-100,100)

    plt.subplot(2,2,3)
    plt.plot(x,sig_21)
    plt.title(title+ ' 21')
    if ref == 1:
        plt.plot([0,0],[-250,max(abs(sig_21))],c='black',linestyle = "dotted")
        if zoom == 1:
            plt.xlim(-100,100)

    plt.subplot(2,2,4)
    plt.plot(x,sig_22)
    plt.title(title+ ' 22')
    
    if ref == 1:
        plt.plot([0,0],[-250,max(abs(sig_22))],c='black',linestyle = "dotted")
        if zoom == 1:
            plt.xlim(-100,100)

    plt.show()

#%%
def real2complex_fir(rx):
    
    if type(rx) == dict:
        h_11_I  = rx['h_est'][0,0,0,:]
        h_12_I  = rx['h_est'][0,1,0,:]
        h_21_I  = rx['h_est'][1,0,0,:]
        h_22_I  = rx['h_est'][1,1,0,:]
    
        h_11_Q  = rx['h_est'][0,0,1,:]
        h_12_Q  = rx['h_est'][0,1,1,:]
        h_21_Q  = rx['h_est'][1,0,1,:]
        h_22_Q  = rx['h_est'][1,1,1,:]
    else:
        h_11_I  = rx[0,0,0,:]
        h_12_I  = rx[0,1,0,:]
        h_21_I  = rx[1,0,0,:]
        h_22_I  = rx[1,1,0,:]
    
        h_11_Q  = rx[0,0,1,:]
        h_12_Q  = rx[0,1,1,:]
        h_21_Q  = rx[1,0,1,:]
        h_22_Q  = rx[1,1,1,:]        
    
    h_11    = h_11_I+1j*h_11_Q
    h_12    = h_12_I+1j*h_12_Q
    h_21    = h_21_I+1j*h_21_Q
    h_22    = h_22_I+1j*h_22_Q
    
    Ntaps                   = max(rx['h_est'].shape)
    rx['h_est_cplx']        = np.zeros((4,Ntaps)).astype(dtype=complex)
    
    rx['h_est_cplx'][0,:]   = h_11.detach().numpy()
    rx['h_est_cplx'][1,:]   = h_12.detach().numpy()
    rx['h_est_cplx'][2,:]   = h_21.detach().numpy()
    rx['h_est_cplx'][3,:]   = h_22.detach().numpy()
    
    rx              = misc.sort_dict_by_keys(rx)
    
    return rx

#%%
def show_fir_central_tap(rx):

    if type(rx) == dict:
        
        if "h_est_cplx" not in rx:
            rx = real2complex_fir(rx)
            
        fir = rx['h_est_cplx']

    else:
        fir = rx
    
    if type(fir) == torch.Tensor:
        fir         = fir.detach().numpy()
        
    loc = round(len(rx['h_est_cplx'].T)/2)
    h   = np.reshape(np.abs(fir[:,loc]),(2,2))
    
    print(h)
        
    
#%%
def show_pol_matrix_central_tap(fibre,rx):

    theta = fibre['thetas'][rx['Frame'],1]
    matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    print(matrix)
    
