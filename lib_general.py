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


import matplotlib.pyplot as plt
import numpy as np
import torch
import lib_misc as misc

pi = np.pi

            # ================================================ #
            # ================================================ #
            # ================================================ #

def plot_const_2pol(sig,*varargin):
    
    Nsps = 2
    
    if np.isrealobj(sig) == 0:
        polH = sig[0]
        polV = sig[1]
        
        polHI = np.real(polH[0::Nsps])
        polHQ = np.imag(polH[0::Nsps])
        polVI = np.real(polV[0::Nsps])
        polVQ = np.imag(polV[0::Nsps])

    else:
        polHI = sig[0][0]
        polHQ = sig[0][1]
        polVI = sig[1][0]
        polVQ = sig[1][1]


    if type(sig) != torch.Tensor:
        polHI = misc.my_tensor(polHI)
        polHQ = misc.my_tensor(polHQ)
        polVI = misc.my_tensor(polVI)
        polVQ = misc.my_tensor(polVQ)
        
    MHI         = torch.mean(torch.abs(polHI)**2)
    MHQ         = torch.mean(torch.abs(polHQ)**2)
    MVI         = torch.mean(torch.abs(polVI)**2)
    MVQ         = torch.mean(torch.abs(polVQ)**2)
    
    MMM         = np.sqrt(max([MHI,MHQ,MVI,MVQ]))
    
    polHInorm   = polHI/MMM
    polHQnorm   = polHQ/MMM
    polVInorm   = polVI/MMM
    polVQnorm   = polVQ/MMM
    
    plt.figure()
    
    plt.subplot(1,2,1)
    plt.scatter(polHInorm[0::Nsps], polHQnorm[0::Nsps],c='black')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel("in phase")
    plt.ylabel("quadrature")
    
    if len(varargin) == 1:
        plt.title('H ' + varargin[0])
    plt.gca().set_aspect('equal', adjustable='box')
        
    plt.subplot(1,2,2)
    plt.scatter(polVInorm[0::Nsps], polVQnorm[0::Nsps],c='black')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel("in phase")
    
    if len(varargin) == 1:
        plt.title('V '+varargin[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    
    

            # ================================================ #
            # ================================================ #
            # ================================================ #

def plot_const_1pol(sig,*varargin):
    
    Nsps = 2
    
    if np.isrealobj(sig) == 0:
        
        polHI = np.real(sig[0::Nsps])
        polHQ = np.imag(sig[0::Nsps])

    else:
        polHI = sig[0][0]
        polHQ = sig[0][1]


    if type(sig) != torch.Tensor:
        polHI = misc.my_tensor(polHI)
        polHQ = misc.my_tensor(polHQ)
        
    MHI         = torch.mean(torch.abs(polHI)**2)
    MHQ         = torch.mean(torch.abs(polHQ)**2)
    
    MMM         = np.sqrt(max([MHI,MHQ]))
    
    polHInorm   = polHI/MMM
    polHQnorm   = polHQ/MMM
    
    plt.figure()
    
    plt.scatter(polHInorm[0::Nsps], polHQnorm[0::Nsps],c='black')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel("in phase")
    plt.ylabel("quadrature")
    
    if len(varargin) == 1:
        plt.title('H ' + varargin[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

            # ================================================ #
            # ================================================ #
            # ================================================ #


            # ================================================ #
            # ================================================ #
            # ================================================ #

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
        
        
            # ================================================ #
            # ================================================ #
            # ================================================ #

def deg2rad(myangle,*varargin):
    
    out_angle = myangle*pi/180
    
    if len(varargin) == 1:
        out_angle = np.round(out_angle,varargin[0])
        
    return out_angle


            # ================================================ #
            # ================================================ #
            # ================================================ #


def rad2deg(myangle,*varargin):

    out_angle = myangle*180/pi
    
    if len(varargin) == 1:
        out_angle = np.round(out_angle,varargin[0])
