# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 1.3.0
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
#   1.0.0 (2024-03-04) - creation
#   1.1.0 (2024-03-28) - [NEW] inverse 3D matrix
#   1.2.1 (2024-04-01) - [NEW] test_unitarity / test_unitarity3d
#   1.2.2 (2024-05-27) - [NEW] get_power
#   1.2.3 (2024-06-18) - [NEW] zero_stuffing, cleaning
#   1.2.4 (2024-06-20) - zero_stuffing: enabling multiple signal processing
#   1.3.0 (2024-06-27) - [NEW] my_low_pass_filter
#   1.3.1 (2024-06-30) - [NEW] mse
#
# ----- MAIN IDEA -----
#   Advanced mathematical operations
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   [A1] Authors        :
#       Title           : 
#       Journal/Editor  : 
#       Volume - N°     : 
#       Date            : 
#       DOI/ISBN        : 
#       Pages           : 
#  ----------------------
#   CODE
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
# fft_matrix
# get_power             (1.2.2)
# inverse3Dmatrix       (1.1.0)
# mse                   (1.3.1)
# my_low_pass_filter    (1.3.0)
# test_unitarity        (1.2.1)
# test_unitarity3d      (1.2.1)
# update_fir            (1.0.0)
# zero_stuffing         (1.2.3)
# =============================================================================

#%%
import numpy as np
import sys
import torch
import lib_matlab as mb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

#%%
def fft_matrix(matrix):
    
    matrixFFT = np.zeros(matrix.shape,dtype = complex)
    for k in range(len(matrix)):
        matrixFFT[k] = np.fft.fftshift(np.fft.fft(matrix[k]))

    return matrixFFT

#%%
def get_power(sig, flag_real2cplx = 0, flag_flatten = 0):

    if type(sig) == torch.Tensor:
        sig     = sig.numpy()
        
    shape       = sig.shape
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

#%%
def inverse3Dmatrix(matrix3D):

    matrix3Dshape   = matrix3D.shape
    matrix3Dinverse = np.zeros(matrix3Dshape)
    Nmatrix2D       = matrix3Dshape[-1]
    
    for k in range(Nmatrix2D):
        matrix3Dinverse[k,:,:] = np.linalg.inv(matrix3D[k,:,:])

    return matrix3Dinverse



#%%
def mse(x,ref):
    
    return np.mean(np.abs(x-ref))



#%% ChatGPT
def my_low_pass_filter(signal, filter_params):
# =============================================================================
# signal         : signal to filter
# filter_params  : changes according to the filter type:
#
# rect_filter_params = {
#     'type'          : 'rectangular',
#     'cutoff'        : cutoff,
#     'fs'            : fs
# }
# 
# butter_filter_params = {
#     'type'          : 'butterworth',
#     'order'         : order,
#     'cutoff'        : cutoff,
#     'fs'            : fs
# }
# 
# ma_filter_params = {
#     'type'          : 'moving_average',
#     'window_size'   : window_size,
#     'average_type'  : 'uniform'
# }
# =============================================================================

    def rectangular_filter(signal, cutoff, fs,):
        n           = len(signal)
        freq        = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))

        fft_signal  = np.fft.fftshift(np.fft.fft(signal))
        filter_mask = np.abs(freq) <= cutoff
        
        fft_signal[~filter_mask] = 0
        filtered_signal = np.fft.ifft(np.fft.fftshift(fft_signal))
        
        # ------------------------------------------------------------ to check
        # plt.figure()
        # plt.semilogy(freq, np.abs(fft_signal), label="signal fft")
        # plt.semilogy(freq, np.abs(filter_mask), label="filter")
        # plt.legend()
        # plt.xlim([-15,15])
        # plt.show()
        # 
        # plt.figure()
        # dt    = 1/fs
        # time  = np.linspace(0, (n-1)*dt, n)
        # plt.plot(time, signal)
        # plt.plot(time, np.real(filtered_signal), label="filtered signal")
        # plt.legend()
        # plt.show()
        # ------------------------------------------------------------ to check
        
        return np.real(filtered_signal)
    # ----------------------------------------------------------------------- #
    
    def butterworth_filter(signal, cutoff, order, fs):
        nyq             = 0.5 * fs
        normal_cutoff   = cutoff / nyq
            
        b, a            = butter(order, normal_cutoff, btype='low')
        y               = filtfilt(b, a, signal)
        
        return y
    # ----------------------------------------------------------------------- #
    
    def moving_average_filter(signal, window_size = 10, average_type='uniform',
                             xmax = 3, *varargin):
        
        if average_type == 'uniform':
            window = np.ones(int(window_size)) / float(window_size)
            
        elif average_type == 'gaussian':
            if len(varargin) != 0:
                std_dev = varargin[0]
            else:
                std_dev = window_size/6
                
            window  = np.exp(-0.5*(np.linspace(-xmax,xmax,window_size)/std_dev) ** 2)
            window  /= np.sum(window)
            
        else:
            raise ValueError("Invalid. Use 'uniform' or 'gaussian'.")
        
        filtered_signal = np.convolve(signal, window, 'same')
        
        return filtered_signal
    # ----------------------------------------------------------------------- #
    
    filter_type = filter_params.get('type', '').lower()
    
    if 'rect' in filter_type.lower():
        return rectangular_filter(signal, filter_params['cutoff'],\
                                  filter_params['fs'])
    
    elif 'butter' in filter_type.lower():
        return butterworth_filter(signal, filter_params['cutoff'],\
                                  filter_params['order'], filter_params['fs'])

    elif 'moving' in filter_type.lower():
        
        if 'uni' in filter_params['ma_type'].lower():
            return moving_average_filter(signal, filter_params['window_size'])
        
        elif 'gauss' in filter_params['ma_type'].lower():
            if "xmax" not in filter_params:
                filter_params['xmax'] = 3
                
            if 'std' not in filter_params:
                filter_params['std'] = filter_params['window_size']/6
                
            return moving_average_filter(signal,
                                         filter_params['window_size'],
                                         "gaussian",
                                         filter_params['xmax'],
                                         filter_params['std'])
    
    else:
        raise ValueError("Invalid. Use 'rectangular', 'butterworth', or\
                         'moving_average'.")



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
def update_fir(loss,optimiser):

    loss.backward()           # gradient calculation
    optimiser.step()          # weights update
    optimiser.zero_grad()     # gradient clearing
    
    


#%% ChatGPT
# example:
    
#     sig_in      = array([1, 2, 3, 4, 5, 6, 7, 8])
#     sig_stuffed = array([[1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0]])
#     sig_stuffed = zero_stuffing(sig_in,Nzeros = 2, Ntimes = 2)


def zero_stuffing(sig_in, Nzeros, Ntimes):

    sig_stuffed = [[] for k in range(len(sig_in))]
    
    for k in range(len(sig_in)):
        Nzeros, Ntimes  = int(Nzeros), int(Ntimes)
        
        # if sig_in.shape[0] == 1:
        #     sig_in = sig_in.squeeze()

        if sig_in[k].shape[0] == 1:
            tmp_sig = sig_in[k].squeeze()
        else:
            tmp_sig = sig_in[k]


            
        tmp         = np.reshape(tmp_sig, (Ntimes, -1))
        my_zeros    = np.zeros((Ntimes, Nzeros))
        
        # Concatenate the zeros to the end of each row
        sig_stuffed[k] = np.hstack((tmp, my_zeros)).reshape((1,-1))

        # ------------------------------------------------------------ to check
        # sig_stuffed = np.reshape((1,-1))
        # ------------------------------------------------------------ to check    
    
    return sig_stuffed