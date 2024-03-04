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
#   Authors             : Vincent LAUINGER
#   Title               : Blind Equalization and Channel Estimation in Coherent Optical
#                         Communications Using Variational Autoencoders
#   Jounal/Editor       : JOURNAL ON SELECTED AREAS IN COMMUNICATIONS
#   Volume - N°         : 40-9
#   Date                : 2022-11
#   DOI/ISBN            : 10.1109/JSAC.2022.3191346
#   Pages               : 2529 - 2539
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


#%%
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time


import lib_kit as kit
import lib_misc as misc
import lib_prop as prop
import lib_txdsp as txdsp
import lib_ai as ai


device = 'cuda' if torch.cuda.is_available() else 'cpu'


#%%
def processing_self(tx,fibre,rx,saving,flags):
        
    tx,fibre,rx = init_processing(tx,fibre,rx,device)

    for frame in range(rx['Nframes']):
        with torch.set_grad_enabled(True):
            rx              = init_train(tx,rx,frame)
            tx              = txdsp.transmitter(tx,rx)
            tx,fibre,rx     = prop.propagation(tx,fibre,rx)
            
            for batchNo in range(rx["NBatchFrame"]):
                rx          = kit.train_self(batchNo,tx,fibre,rx,flags)
                rx,loss     = kit.loss_function_shaping(tx,rx)
                ai.optimise(loss,rx['optimiser'])
                
            plot_loss_batch(rx,flags,saving,['kind','law',"std",'linewidth'],skip = 5)

        
        rx["H_est_l"].append(rx["h_est"].tolist())

        
        rx      = kit.SNR_estimation(tx,rx)
        rx      = kit.SER_estimation(tx,rx,rx['out_train'],frame)
        array   = print_results(loss,frame,fibre,rx,saving)
        
    # END ------- for frame in range(rx['Nframes']):
    tx,fibre, rx = save_data(tx,fibre,rx,array,saving)

    return tx,fibre,rx    








####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


#%%
# ============================================================================================ #
# ============================================================================================ #
def print_results(loss,frame,fibre,rx,saving):
    
    SER_valid   = rx["SER_valid"]
    SERs        = rx["SERs"]
    Losses      = rx["Losses"]
    SNRdB_est   = rx["SNRdB_est"][frame].item()  
    SNRdBs      = rx["SNRdBs"]

    lossk       = round(loss.item(),0)
    
    SERvalid0   = round(SER_valid[0,frame].item(),10)
    SERvalid1   = round(SER_valid[1,frame].item(),10)
    SERvalid2   = round(SER_valid[2,frame].item(),10)
    SERvalid3   = round(SER_valid[3,frame].item(),10)
    
    SERsvalid   = np.array([SERvalid0,SERvalid1,SERvalid2,SERvalid3])
    SERmeank    = round(np.mean(SERsvalid),10)

    
    Shifts      = np.array([rx['shift'][0].item(),rx['shift'][1].item()])
    Shiftsmean = round(np.mean(Shifts),1)
    
    SNRdBk     = round(10*np.log10(SNRdB_est),5)
        
    SERs.append(SERmeank)
    Losses.append(lossk)
    SNRdBs.append(SNRdBk)
    
    Iteration   = misc.linspace(1,len(Losses),len(Losses))
    Losses      = misc.list2vector(Losses)
    SNRdBs      = misc.list2vector(SNRdBs)
    SERs        = misc.list2vector(SERs)
    array       = np.concatenate((Iteration,Losses,SNRdBs,SERs),axis=1)
    thetak      = fibre['thetas'][frame][1]+fibre['thetas'][frame][0]
        
    print("frame %d"     % frame,
      '--- loss = %.1f'      % lossk,
      '--- SNRdB = %.2f'    % SNRdBk,
      '--- Theta = %.2f'    % (thetak*180/np.pi),
      '--- <SER> = %.2e'     % SERmeank,
      '--- <shift> = %.1f'   % Shiftsmean,
      '--- r = %i'           % rx['r']
      )
    
    
    
    '''
        shift_x = %.1f' % shift[0].item()
        shift_y = %.1f' % shift[1].item(),
        
        (constell. with shaping)
        SER_x = %.3e '  % SERvalid0,
        SER_y = %.3e'   % SERvalid1,
        
        (soft demapper)
        SER_x = %.3e '  % SERvalid2,
        SER_y = %.3e'   % SERvalid3,
    '''
    
    return array
    



#%%
# ============================================================================================ #
# ============================================================================================ #

def save_data(tx,fibre,rx,array,saving):
    
    
    Thetas_IN   = np.array([fibre["thetas"][k][0] for k in range(rx["Nframes"])])
    Thetas_OUT  = np.array([fibre["thetas"][k][1] for k in range(rx["Nframes"])])
    thetas      = list((Thetas_IN+Thetas_OUT)*180/np.pi)
    Thetas      = misc.list2vector(thetas)
    array       = np.concatenate((array,Thetas),axis=1)

    
    misc.array2csv(array, saving["filename"],["iteration","loss","SNR","SER","Thetas"])

    tx          = misc.sort_dict_by_keys(tx)
    fibre       = misc.sort_dict_by_keys(fibre)            
    rx          = misc.sort_dict_by_keys(rx)

    return tx,fibre,rx







#%%
# ============================================================================================ #
# intialisation of the channel as the identity (in our case (h0)) and modulation format
# ============================================================================================ #
# as for now, there is no PMD, CD : it will be added in the generate_data_shaping function
# Bibliography: 
# [1] Caciularu et al. ICC 2018, 10.1109/ICCW.2018.8403666
# [2] You et al. TNN Vol9 No6 1998
#
# louis tomczyk modif: 2023-08-20

def init_processing(tx,fibre,rx,device):
    tx["Npolars"]           = 2
    tx["Ntaps"]             = tx["Nsps"]*tx["NSymbTaps"]-1
    
    # Channel matrix initiatlisation: Dirac
    h_est                           = np.zeros([tx['Npolars'],tx['Npolars'],2,tx["Ntaps"]])
    h_est[0,0,0,tx["NSymbTaps"]-1]  = 1
    h_est[1,1,0,tx["NSymbTaps"]-1]  = 1
    h_est                           = misc.my_tensor(h_est,requires_grad=True)
 
    tx, rx                  = txdsp.get_constellation(tx,rx)
    SNR                     = 10**(rx["SNRdB"]/10)
    rx["noise_var"]         = torch.full((2,),tx['pow_mean']/SNR/2)
    
    tx["fs"]                = tx["Rs"]*tx["Nsps"]           # [GHz]
    
    rx["h_est"]             = h_est
    rx['BatchLen']          = np.floor(rx['NSymbFrame']/rx["NBatchFrame"]).astype("int")
    rx["NsampBatch"]        = rx['BatchLen']* tx['Nsps']
    
    rx["N_cut"]             = 10                            # number of symbols cut off to prevent edge effects of convolution
    rx["Nsamp_rx"]          = tx["Nsps"]*rx["NSymbFrame"]
    
    
    fibre                   = prop.set_thetas(tx,fibre,rx)  # polarisations rotation angles generation
    
    rx['net']               = kit.twoXtwoFIR(tx).to(device)
    rx['optimiser']         = optim.Adam(rx['net'].parameters(), lr = rx['lr'])
    rx['optimiser'].add_param_group({"params": rx["h_est"]}) 
    
    # initialisation of the outputs    
    rx["H_est_l"]           = []
    rx["H_lins"]            = []
    
    rx["Losses"]            = []
    rx["SNRdBs"]            = []
    rx["SNRdB_est"]         = np.zeros(rx['Nframes'])
    rx["SER_valid"]         = []
    rx["SERs"]              = []
    
    rx["SER_valid"]         = misc.my_zeros_tensor((4,rx['Nframes']))
    rx['Pnoise_est']        = misc.my_zeros_tensor((tx["Npolars"],rx['Nframes']))
    rx["minibatch_real"]    = misc.my_zeros_tensor((tx["Npolars"],2,rx["NsampBatch"]))
    
    

    tx      = misc.sort_dict_by_keys(tx)
    fibre   = misc.sort_dict_by_keys(fibre)
    rx      = misc.sort_dict_by_keys(rx)
    
    return tx,fibre,rx



#%%
# ============================================================================================ #
# ============================================================================================ #
def plot_loss_batch(rx,flags,saving,keyword,skip):
    
    # a tensor with GRAD on cannot be plotted, it requires first to put it in a normal array
    # that's the purpose of DETACH
    
    # do not intervert SAVEFIG and SHOW otherwise the figure will not be saved

    if flags['plot_loss_batch']:

        x = [k for k in range(rx['batchNo'])]
        y = rx['losses_subframe'][rx['Frame']][:rx['batchNo']].detach().numpy()        
        
        if rx['Frame'] >= rx['FrameRndRot']:
            linestyle   = "solid"
        else:
            linestyle = ":"
         
        if rx['Frame'] == rx['FrameRndRot']:
            plt.plot(x,y,linestyle=linestyle,linewidth = 5, color = 'k',label = 'frame {} - after channel 1st round'.format(rx["Frame"]))   
         
        elif rx['Frame']%skip == 0:
            if rx["Frame"]<rx['FrameRndRot']:
                plt.plot(x,y,linestyle=linestyle,label = 'frame {} - b4 channel'.format(rx["Frame"]))
            else:
                plt.plot(x,y,linestyle=linestyle,label = 'frame {} - after channel'.format(rx["Frame"]))

        if rx['Frame'] == rx['Nframes']-1:
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.ylim([-200,350])

            plt.legend()
            
            
            if keyword != []:
                patterns        = []
                pattern_values  = []
                
                for kw in keyword:
                    pattern_index   = misc.find_string(kw,saving['filename'])
                    patterns.append(saving['filename'][pattern_index[0]:pattern_index[1]])
                    pattern_values.append(saving['filename'][pattern_index[1]+1:pattern_index[1]+1+2])
                
                tname = 'loss in batches - '
                for k in range(len(patterns)):
                    tname = tname + "{}-{} -- ".format(patterns[k],pattern_values[k])
                    
                plt.title(tname)
            # if keyword != []:
            #     pattern_index   = misc.find_string(keyword,saving['filename'])
            #     pattern         = saving['filename'][pattern_index[0]:pattern_index[1]]
            #     pattern_value   = saving['filename'][pattern_index[1]+1:pattern_index[1]+1+2]
                            
                # plt.title("loss in the batches - {} = {}".format(pattern,pattern_value))
            else:
                plt.title("loss in the batches")
                
            output_file = "{}.png".format(saving['filename']+' loss_batch')
            plt.savefig(output_file,bbox_inches='tight')
            plt.show()


#%%
# ============================================================================================ #
# ============================================================================================ #

def init_train(tx,rx,frame):
    
    if rx["N_lrhalf"] > rx["Nframes"]:
        
        rx["lr_scheduled"]              = rx["lr"] * 0.5
        rx['optimiser'].param_groups[0]['lr'] = rx["lr_scheduled"]
        
    rx['out_train']         = misc.my_zeros_tensor((tx["Npolars"],2*tx["N_amps"],rx["NSymbFrame"]))
    rx['out_const']         = misc.my_zeros_tensor((tx["Npolars"],2,rx["NSymbFrame"]))
    rx['Pnoise_batches']    = misc.my_zeros_tensor((tx["Npolars"],rx["NBatchFrame"]))
    rx["Frame"]             = frame
    rx['losses_subframe']   = misc.my_zeros_tensor((rx["Nframes"],rx['NBatchFrame']))
    
    rx['net'].train()
    rx['optimiser'].zero_grad()
    
    return rx