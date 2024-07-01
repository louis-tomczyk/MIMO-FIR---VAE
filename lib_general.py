# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.5.2
#   Date            : 2024-07-01
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
#   1.1.1 (2024-04-01) - [NEW] plot_fir
#   1.2.0 (2024-04-03) - plot_const_2pol
#                      - [NEW] show_fir_central_tap
#   1.2.1 (2024-04-07) - plot_const_1d, plot_const_2d
#   1.3.0 (2024-05-21) - [NEW] plot_decisions, plot_xcorr_2x2
#   1.3.1 (2024-05-21) - plot_loss_cma --- y = -y if mean(y[0:10])<0
#   1.4.0 (2024-05-24) - plot_const_1pol --- cplx to real processing
#                      - [NEW] plot_const_2pol_2sig
#   1.4.1 (2024-05-24) - plot_const_2pol_2sig, plot_const_1/2pol (Npoints max)
#   1.5.0 (2024-06-06) - Naps -> NsampTaps, plot_const_1/2pol
#                      - [REMOVED] plot_const_1/2pol(_2sig), plot_phases
#                      - [NEW] plot_constellations, to replace the removed ones
#   1.5.1 (2024-06-20) - plot_constellations: adding the grid
#   1.5.2 (2024-07-01) - real2complex_fir: torch.Tensor type management
# 
# ----- MAIN IDEA -----
#   Library for plotting functions in (optical) telecommunications
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
#  ----------------------
#   CODE
#   [C1] Author         : Vincent Lauinger
#        Contact        : vincent.lauinger@kit.edu
#        Laboratory/team: Communications Engineering Lab
#        Institution    : Karlsruhe Institute of Technology (KIT)
#        Date           : 2022-06-15
#        Program Title  : 
#        Code Version   : 
#        Web Address    : https://github.com/kit-cel/vae-equalizer
#   [C2] Authors        : Jingtian Liu, Élie Awwad, louis tomczyk
#       Contact         : elie.awwad@telecom-paris.fr
#       Affiliation     : Télécom Paris, COMELEC, GTO
#       Date            : 2024-04-27
#       Program Title   : 
#       Code Version    : 3.0
#       Type            : Source code
#       Web Address     : 
# ---------------------------------------------


#%% =============================================================================
# --- CONTENTS ---
# - fir_2Dto3D
# - fir_3Dto2D
# - plot_constellations     (1.5.0)
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
    NsampTaps       = len(fir.transpose())
    tmp         = np.zeros((2,2,NsampTaps),dtype = complex)
    
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
    NsampTaps       = fir_shape[-1]
    tmp         = np.zeros(4,NsampTaps)
    
    tmp[0,:]    = fir[0,0,:]
    tmp[1,:]    = fir[0,1,:]
    tmp[2,:]    = fir[1,0,:]
    tmp[3,:]    = fir[1,1,:]
    
    return tmp



#%%
def plot_constellations(sig1, sig2=None, labels=None, norm=0, sps=2, polar='H', title='', axislim=[-2, 2]):
    def process_signal(sig):
        if type(sig) == torch.Tensor:
            sig = sig.detach().numpy()
            
        if np.isrealobj(sig) == 0:
            polHI = np.real(sig[0]).flatten()
            polHQ = np.imag(sig[0]).flatten()
            polVI = np.real(sig[1]).flatten() if len(sig) > 1 else None
            polVQ = np.imag(sig[1]).flatten() if len(sig) > 1 else None
            
        else:
            if sig.shape[0] != 2:
                polHI = sig[0].flatten()
                polHQ = sig[1].flatten()
                polVI = sig[2].flatten() if sig.shape[0] > 2 else None
                polVQ = sig[3].flatten() if sig.shape[0] > 2 else None

            else:
                polHI = sig[0][0].flatten()
                polHQ = sig[0][1].flatten()
                polVI = sig[1][0].flatten() if sig.shape[0] > 1 else None
                polVQ = sig[1][1].flatten() if sig.shape[0] > 1 else None

        return polHI, polHQ, polVI, polVQ       
    # ----------------------------------------------------------------------- #

    def truncate_and_normalize(polHI, polHQ, polVI, polVQ):
        Nmax = int(5e3)
        if polHI is not None and len(polHI) > Nmax:
            polHI   = polHI[:Nmax]
            polHQ   = polHQ[:Nmax]
            
            if polVI is not None:
                polVI = polVI[:Nmax]
                polVQ = polVQ[:Nmax]

        if norm == 1:
            M1HI    = np.mean(abs(polHI)**2)
            M1HQ    = np.mean(abs(polHQ)**2)
            M1VI    = np.mean(abs(polVI)**2) if polVI is not None else 0
            M1VQ    = np.mean(abs(polVQ)**2) if polVQ is not None else 0
            MMM     = np.sqrt(max([M1HI, M1HQ, M1VI, M1VQ]))
            
        else:
            MMM = 1

        polHInorm = polHI / MMM
        polHQnorm = polHQ / MMM
        polVInorm = polVI / MMM if polVI is not None else None
        polVQnorm = polVQ / MMM if polVQ is not None else None
        
        return polHInorm, polHQnorm, polVInorm, polVQnorm
    # ----------------------------------------------------------------------- #

    def plot_subplot(subplot_idx, polar, polHInorm, polHQnorm, polVInorm, polVQnorm, color, label):
        fontsize = 12
        plt.subplot(1, 2, subplot_idx)
        
        eps = np.finfo(float).eps
        x   = np.linspace(-1+eps,1-eps,75)
        y1p = np.sqrt(1-x**2)
        y1m = -np.sqrt(1-x**2)
        
        
        plt.plot([-1, 1], [0, 0],  c='black', linestyle='dotted', linewidth = 0.5)
        plt.plot([0, 0],  [-1, 1], c='black', linestyle='dotted', linewidth = 0.5)
        plt.plot([-1, 1], [-1, 1], c='black', linestyle='dotted', linewidth = 0.5)
        plt.plot([-1, 1], [1, -1], c='black', linestyle='dotted', linewidth = 0.5)
        
        plt.plot(x, y1p, c='black', linestyle='dotted', linewidth = 0.5)
        plt.plot(x, y1m, c='black', linestyle='dotted', linewidth = 0.5)
        


        if polar.lower() == 'h':
            plt.scatter(polHInorm[0::sps], polHQnorm[0::sps], c=color, label=label)
            plt.xlabel("In Phase", fontsize=fontsize)
            plt.ylabel("Quadrature", fontsize=fontsize)
            if title.lower() != '':
                plt.title('pol1 ' + title, fontsize=fontsize)
        elif polar.lower() == 'v' and polVInorm is not None and polVQnorm is not None:
            plt.scatter(polVInorm[0::sps], polVQnorm[0::sps], c=color, label=label)
            plt.xlabel("In Phase", fontsize=fontsize)
            plt.ylabel("Quadrature", fontsize=fontsize)
            if title.lower() != '':
                plt.title('pol2 ' + title, fontsize=fontsize)
                
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(axislim)
        plt.ylim(axislim)

    # ----------------------------------------------------------------------- #

    sig1_data = process_signal(sig1)
    sig2_data = process_signal(sig2) if sig2 is not None else (None, None, None, None)

    pol1HInorm, pol1HQnorm, pol1VInorm, pol1VQnorm = truncate_and_normalize(*sig1_data)
    pol2HInorm, pol2HQnorm, pol2VInorm, pol2VQnorm = truncate_and_normalize(*sig2_data) if sig2 is not None else (None, None, None, None)

    plt.figure()

    if polar.lower() == 'both':
        plot_subplot(1, 'h', pol1HInorm, pol1HQnorm, pol1VInorm, pol1VQnorm, 'black', labels[0] if labels else None)
        if sig2 is not None:
            plot_subplot(1, 'h', pol2HInorm, pol2HQnorm, pol2VInorm, pol2VQnorm, 'blue', labels[1] if labels else None)

        plot_subplot(2, 'v', pol1HInorm, pol1HQnorm, pol1VInorm, pol1VQnorm, 'black', labels[0] if labels else None)
        if sig2 is not None:
            plot_subplot(2, 'v', pol2HInorm, pol2HQnorm, pol2VInorm, pol2VQnorm, 'blue', labels[1] if labels else None)
    else:
        plot_subplot(1, polar, pol1HInorm, pol1HQnorm, pol1VInorm, pol1VQnorm, 'black', labels[0] if labels else None)
        if sig2 is not None:
            plot_subplot(1, polar, pol2HInorm, pol2HQnorm, pol2VInorm, pol2VQnorm, 'blue', labels[1] if labels else None)

    if labels:
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
        NsampTaps = max(rx['h_est'].shape)

        if len(rx["h_est"].shape) == 4:
            rx      = real2complex_fir(rx)
            myfir   = abs(rx['h_est_cplx'])

        else:
            myfir   = abs(rx["h_est"])

    else:
        myfir2 = np.abs(rx)
        if len(rx.shape) == 1:
            NsampTaps   = len(rx.shape)
            myfir       = np.zeros((4,NsampTaps))
            myfir[0,:]  = myfir2
            myfir[3,:]  = myfir2            
        else:
            NsampTaps   = max(myfir2.shape)
            myfir       = myfir2


    Taps = np.linspace(1,NsampTaps,NsampTaps)-round(NsampTaps/2)-1
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

def plot_loss_cma(rx,saving,keyword,what):
    
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

def plot_xcorr_2x2(sig_11,sig_12,sig_21, sig_22,title,ref=0,zoom=0):

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
        rx_h_est = rx['h_est']
        
    if type(rx_h_est) == torch.Tensor:
        rx_h_est = rx_h_est.detach().numpy()
        
    if type(rx) == dict:
        
        h_11_I  = rx_h_est[0,0,0,:]
        h_12_I  = rx_h_est[0,1,0,:]
        h_21_I  = rx_h_est[1,0,0,:]
        h_22_I  = rx_h_est[1,1,0,:]
    
        h_11_Q  = rx_h_est[0,0,1,:]
        h_12_Q  = rx_h_est[0,1,1,:]
        h_21_Q  = rx_h_est[1,0,1,:]
        h_22_Q  = rx_h_est[1,1,1,:]
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
    
    NsampTaps               = max(rx['h_est'].shape)
    rx['h_est_cplx']        = np.zeros((4,NsampTaps)).astype(dtype=complex)
    
    rx['h_est_cplx'][0,:]   = h_11
    rx['h_est_cplx'][1,:]   = h_12
    rx['h_est_cplx'][2,:]   = h_21
    rx['h_est_cplx'][3,:]   = h_22
    
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
    