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

import torch
import matplotlib.pyplot as plt
import lib_kit as kit
import lib_ai as ai
import lib_general as gen
import lib_misc as misc

#%%
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def mimo(tx,rx,saving,flags):
    if rx['mimo'].lower() == "vae":

        for batchNo in range(rx["NBatchFrame"]):

            rx              = kit.train_self(batchNo,rx,tx)
            rx,loss         = kit.loss_function_shaping(tx,rx)
            ai.optimise(loss,rx['optimiser'])
                    
        plot_loss_batch(rx,flags,saving,['kind','law',"std",'linewidth'],"Llikelihood")
        plot_loss_batch(rx,flags,saving,['kind','law',"std",'linewidth'],"DKL")
        plot_loss_batch(rx,flags,saving,['kind','law',"std",'linewidth'],"losses")
        gen.plot_const_2pol(rx['out_const'])
        
    else:
        if rx["Frame"]>rx['FrameRndRot']-1:
            rx,loss = kit.CMA(tx,rx)
            plot_loss_cma(rx,flags,saving,['kind','law',"std",'linewidth'],"x")
        else:
            loss = []
        
    return rx,loss

#%%
# ============================================================================================ #
# ============================================================================================ #

def SNR_estimation(tx,rx):

    if rx["mimo"].lower() == "vae":
        rx["SNRdB_est"][rx['Frame']]      = tx["pow_mean"]/torch.mean(rx['Pnoise_batches'])
        rx['Pnoise_est'][:,rx['Frame']]   = torch.mean(rx['Pnoise_batches'],dim=1)  

    return rx
        


#%%
# ============================================================================================ #
# ============================================================================================ #
def plot_loss_cma(rx,flags,saving,keyword,what):
    
    # a tensor with GRAD on cannot be plotted, it requires first to put it in a normal array
    # that's the purpose of DETACH
    
    # do not intervert SAVEFIG and SHOW otherwise the figure will not be saved

    if flags['plot_loss_batch']:

        if what.lower() == "x" or what.lower() == "polX":
            ind = 0
        elif what.lower() == "y" or what.lower() == "polY":
            ind = 1
        else:
            ind = [1,2]
            
        plt.figure(0)

        Nplot = int(rx['NSymbFrame']/5)
        
        x = misc.linspace(0, Nplot-1, Nplot,axis='col')
        y = rx["CMA"]["losses"][str(rx['Frame'])].detach().numpy().transpose()
        y = y[ind][:Nplot].transpose()
        
        if rx['Frame'] >= rx['FrameRndRot']:
            linestyle   = "solid"
        else:
            linestyle = ":"
         
        if rx['Frame'] == rx['FrameRndRot']:
            plt.plot(x,y,linestyle=linestyle,linewidth = 5, color = 'k',label = 'frame {} - after channel 1st round'.format(rx["Frame"]))   
         
        elif rx['Frame']%saving['skip'] == 0:
            if rx["Frame"]<rx['FrameRndRot']:
                plt.plot(x,y,linestyle=linestyle,label = 'frame {} - b4 channel'.format(rx["Frame"]))
            else:
                plt.plot(x,y,linestyle=linestyle,label = 'frame {} - after channel'.format(rx["Frame"]))

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
                
            plt.ylim((-2,2))
            output_file = "{}.png".format(saving['filename']+ ' '+what+'_batch')
            plt.savefig(output_file,bbox_inches='tight')


    
#%%
# ============================================================================================ #
# ============================================================================================ #
def plot_loss_batch(rx,flags,saving,keyword,what):
    
    # a tensor with GRAD on cannot be plotted, it requires first to put it in a normal array
    # that's the purpose of DETACH
    
    # do not intervert SAVEFIG and SHOW otherwise the figure will not be saved

    if flags['plot_loss_batch']:

        Binary  = misc.string_to_binary(what)
        Decimal = misc.binary_to_decimal(Binary)
        Fig_id  = int(Decimal*1e-12)
        plt.figure(Fig_id)
        
        x = [k for k in range(rx['batchNo'])]
        y = rx[what+'_subframe'][rx['Frame']][:rx['batchNo']].detach().numpy()        
        
        if rx['Frame'] >= rx['FrameRndRot']:
            linestyle   = "solid"
        else:
            linestyle = ":"
         
        if rx['Frame'] == rx['FrameRndRot']:
            plt.plot(x,y,linestyle=linestyle,linewidth = 5, color = 'k',label = 'frame {} - after channel 1st round'.format(rx["Frame"]))   
         
        elif rx['Frame']%saving['skip'] == 0:
            if rx["Frame"]<rx['FrameRndRot']:
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
                
            output_file = "{}.png".format(saving['filename']+ ' '+what+'_batch')
            plt.savefig(output_file,bbox_inches='tight')
