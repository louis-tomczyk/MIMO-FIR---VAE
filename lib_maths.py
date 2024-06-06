# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : Louis Tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.2.2
#   Date            : 2024-05-27
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute - place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source - include copyright - state changes - include license

# ----- CHANGELOG -----
#   1.0.0 (2024-03-04) - creation
#   1.1.0 (2024-03-28) - [NEW] inverse 3D matrix
#   1.2.1 (2024-04-01) - [NEW] test_unitarity / test_unitarity3d
#   1.2.2 (2024-05-27) - [NEW] get_power

# ----- MAIN IDEA -----
#   This module provides utilities for matrix inversion and testing unitarity.

# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   [A1] Authors        : 
#       Title           : 
#       Journal/Editor  : 
#       Volume - N°     : 
#       Date            : 
#       DOI/ISBN        : 
#       Pages           : 

#   Functions:
#   [C1] Author         : 
#       Contact         : 
#       Affiliation     : 
#       Date            : 
#       Program Title   : 
#       Code Version    : 
#       Web Address     : 
# ---------------------------------------------
# %%


#%%
# =============================================================================
# update_fir
# inverse3Dmatrix
# fft_matrix
# test_unitarity
# test_unitarity3d
# =============================================================================

#%%
import numpy as np
import sys
import torch
import lib_matlab as mb

#%%
def update_fir(loss,optimiser):

    loss.backward()           # gradient calculation
    optimiser.step()          # weights update
    optimiser.zero_grad()     # gradient clearing
    
    
#%%

def inverse3Dmatrix(matrix3D):

    matrix3Dshape   = matrix3D.shape
    matrix3Dinverse = np.zeros(matrix3Dshape)
    Nmatrix2D       = matrix3Dshape[-1]
    
    for k in range(Nmatrix2D):
        matrix3Dinverse[k,:,:] = np.linalg.inv(matrix3D[k,:,:])

    return matrix3Dinverse


#%%

def fft_matrix(matrix):
    
    matrixFFT = np.zeros(matrix.shape,dtype = complex)
    for k in range(len(matrix)):
        matrixFFT[k] = np.fft.fftshift(np.fft.fft(matrix[k]))

    return matrixFFT




#%%
def test_unitarity(matrix):
    
    matrixDagger    = np.transpose(np.conj(matrix))
    prod1           = matrixDagger*matrix
    prod2           = matrix*matrixDagger
    
    # the matrix is 2x2
    bool1 = np.sum(prod1 == np.eye(2)) == 4
    bool2 = np.sum(prod2 == np.eye(2)) == 4
    
    bool3 = bool1+bool2
    
    return bool3

#%%
def test_unitarity3d(matrix3D):
    
    matrix3Dshape   = matrix3D.shape
    Nmatrix2D       = matrix3Dshape[-1]
    
    result = 0
    for k in range(Nmatrix2D):
        print(matrix3D[:,:,k])
        result = test_unitarity(matrix3D[:,:,k])
        if result != True:
            sys.exit()
        
    return result



#%%
def get_power(sig, flag_real2cplx = 0, flag_flatten = 0):

    if type(sig) == torch.Tensor:
        sig     = sig.numpy()
        
    shape   = sig.shape
    if len(shape) == 1:
        sig     = np.expand_dims(sig,0)
        shape   = sig.shape 



    if flag_real2cplx == 1:
        Npow    = int(shape[0]/2)
        out_pow = np.zeros((Npow,1))
        sigtmp  = np.zeros((Npow,mb.numel(sig[0])))
        sigcplx = np.zeros((Npow,mb.numel(sig[0])),dtype=np.complex64)
        
        if flag_flatten == 1:

            for k in range(Npow):
                sigtmp[0,:]     = np.reshape(sig[2*k],(-1,1))[:,0]  
                sigtmp[1,:]     = np.reshape(sig[2*k+1],(-1,1))[:,0]

                sigcplx[k,:]    = sigtmp[0]+1j*sigtmp[1]
                out_pow[k]      = np.mean(abs((sigcplx[k,:])**2))
        else:
            for k in range(Npow):
                sigtmp[0,:]     = sig[2*k]
                sigtmp[1,:]     = sig[2*k+1]
                
                sigcplx[k,:]    = sigtmp[0]+1j*sigtmp[1]
                out_pow[k]      = np.mean(abs((sigcplx[k,:])**2))
                
    else:
        
        Npow    = shape[0]
        out_pow = np.zeros((Npow,1))
        
        for k in range(Npow):
            out_pow[k]      = np.mean(abs((sig[k,:])**2))

    return out_pow