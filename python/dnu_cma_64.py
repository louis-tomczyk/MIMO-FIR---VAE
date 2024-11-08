# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Version         : 3.0.1
#   Date            : 2024-11-04
#   License         : GNU GPLv2
#                       CAN:    commercial use - modify - distribute
#                               place warranty
#                       CANNOT: sublicense - hold liable
#                       MUST:   include original - disclose source
#                               include copyright - state changes
#                               include license
#
# ----- CHANGELOG -----
#   1.0.0 (2023-03-13) - phase noise added
#   1.1.1 (2023-04-02) - phase/pol dynamics
#   1.2.0 (2023-04-19) - residual chromatic dispersion, Fpol Law Linear
#   1.3.0 (2024-05-27) - cleaning, pcs table
#   1.3.1 (2024-06-04) - blind/pilot aided
#   1.3.2 (2024-06-07) - [DRAFT] blind/pilot aided, cleaning
#   1.3.2 (2024-06-14) - cazac
#   1.4.0 (2024-06-27) - folder management: no more temp folders
#                        instead create_xml_file adds the realisation number
#   1.4.1 (2024-06-30) - phase noise filter parameters
#   1.4.2 (2024-07-02) - phase noise adaptative filter parameters + correction
#   1.4.3 (2024-07-10) - naming normalisation (*frame*-> *Frame*)
#   1.4.4 (2024-07-11) - file naming managed in 'create_xml_file', along with
#                           misc (1.4.0)
# ---------------------
#   2.0.0 (2024-07-12) - LIBRARY NAME CHANGED: LIB_GENERAL -> LIB_PLOT
#   2.0.1 (2024-07-17) - saving elapsed time
#   2.0.2 (2024-07-25) - server management
#   2.0.3 (2024-09-28) - server: avoid plotting
#   2.0.4 (2024-10-03) - seed management for random generation, along with
#                           processing (1.3.10)
#   2.0.5b (2024-10-06)- SoP fpol modification + CFO
#   2.0.6b (2024-10-07) - SoP fpol modificaton + CFO + phase noise: along
#                           txhw (1.2.5)
#   2.0.7b (2024-10-09) - SoP fpol modificaton + CFO + phase noise: along
#                           txhw (1.2.5)
#   2.1.0  (2024-10-10) - reordering the parameters
#   2.2.0  (2024-10-11) - try-except + xml generation, along misc (2.1.3)
#   2.2.1  (2024-10-13) - fail_log
#   2.2.2  (2024-10-15) - moving to parameters_explanations.yaml params
#                           details (entropy tables, pilots management,..)
#   2.2.3  (2024-10-21) - right relationship between fpol/vsop
#   2.2.4  (2024-10-26) - NframesTrain optional
#   2.3.0  (2024-10-31) - managing the NSbB with respect to SNR, Rs
#                       - NFramesChannel with respect to Rs, rxmimo, such that
#                           exepcted abs angle unchanged
# ---------------------
#   3.0.0  (2024-11-03) - changing the slightly the structure: Number of frames
#                           set by the targeted final angle/phase, along with
#                           txdsp (2.4.0)
#   3.0.1  (2024-11-04) - Wrong Nf_Th for Vsop + cleaning
#
# ----- MAIN IDEA -----
#   Simulation of an end-to-end linear optical telecommunication system
#
# ----- BIBLIOGRAPHY -----
#   Articles/Books:
#   [A1] Authors         : Vincent Lauinger
#        Title           : Blind Equalization and Channel Estimation in
#                          Coherent Optical Communications Using Variational
#                          Autoencoders
#        Journal/Editor  : J-SAC
#        Volume - N°     : 40-9
#        Date            : 2022-11
#        DOI/ISBN        : 10.1109/JSAC.2022.3191346
#        Pages           : 2529 - 2539
#  ----------------------
#   CODE
#   [C1] Author          : Vincent Lauinger
#        Contact         : vincent.lauinger@kit.edu
#        Laboratory/team : Communications Engineering Lab
#        Institution     : Karlsruhe Institute of Technology (KIT)
#        Date            : 2022-06-15
#        Program Title   : 
#        Code Version    : 
#        Type            : Source code
#        Web Address     : https://github.com/kit-cel/vae-equalizer
# ---------------------------------------------


#%% ===========================================================================
# --- LIBRARIES ---
# =============================================================================
import numpy            as np
import lib_misc         as misc
import lib_plot         as plot


from processing         import processing
from lib_txdsp          import set_Nsymbols
from lib_txdsp          import set_Batches_Frames
from lib_misc           import show_dict

from lib_matlab         import PWD
from lib_matlab         import clc
from torch              import cuda
from datetime           import date
from random             import randint

from numpy              import floor
from numpy              import sqrt

pi      = np.pi
clc()
np.set_printoptions(linewidth=160)


if "infres" in PWD(show = False).lower():
    server  = 1                                                                   # if = 1, cancels all the checks
else:
    server  = 0

gen_xml     = 0
sort_files  = 1
get_time    = 0


#%% ===========================================================================
# --- MAIN parameters
# =============================================================================
Nrea                    = 10
paramSNR                = [22]          # {21-cma, 17-vae}
Rs                      = 64e9          # [Baud] Symbol rate
rxmimo                  = "cma"         # {cma, vae}
rxmode                  = "pilots"      # {blind, pilots}

CFO_or_dnu              = 'dnu'         # {CFO, dnu, none}
SoPlin_or_fpol          = 'soplin'        # {SoPlin, fpol, none}
Nf_lim                  = 'ph-dnu'      # {ph-dnu, ph-cfo, th-lin, th-fpol}
# NFramesChannel          = 50

# paramNSbB               = [250]
# NframesTrain            = 10
# NSbF                    = 1e4
# NSbB                    = 250


# -----------------------------------------------------------------------------
if SoPlin_or_fpol.lower() == 'fpol' or SoPlin_or_fpol.lower() == 'vsop':
    Vsop        = np.array([5])*1e4                               # [rad/s]
    Th_End      = 5e-2              # (NF > x) <=> Th_End > sqrt(2x/pi) * NSbF*V/Rs
    SoPlin      = np.array([1])
    plot_th     = 1
    
elif SoPlin_or_fpol.lower() == 'soplin':
    SoPlin      = np.array([5])*1e4
    Th_End      = 0
    Vsop        = np.array([1])
    plot_th     = 1
    
elif SoPlin_or_fpol.lower() == 'none':
    SoPlin       = np.array([1])
    Th_End      = 0
    Vsop        = np.array([1])
    plot_th     = 0

# -----------------------------------------------------------------------------
if CFO_or_dnu.lower() == 'dnu':
    Dnus        = np.array([0.5,1,2,5,10])*1e4                               # [Hz]
    Ph_End      = 90*pi/180
    CFO         = np.array([1])
    plot_ph     = 1

elif CFO_or_dnu.lower() == 'cfo':
    CFO         = np.array([1])*1e4
    Ph_End      = pi/2
    Dnus        = np.array([1])
    plot_ph     = 1
    
elif CFO_or_dnu.lower() == "none":
    CFO         = np.array([1])
    Ph_End      = 0
    Dnus        = np.array([0.1])
    plot_ph     = 0
# -----------------------------------------------------------------------------
# keep [1] to avoid divisions by zero





#%% ===========================================================================
# --- PARAMETERS ---
# =============================================================================

Date                     = date.today().strftime("%Y-%m-%d")
tx,fibre,rx,saving       = misc.init_dict(server)

###############################################################################
################################ TRANSMITTER ##################################
###############################################################################

tx['SNRdB']             = 50
tx['Rs']                = Rs


tx["mod"]               = '64QAM' # {4,16,64}QAM
tx["nu"]                = 0.0254  # for PCS: exp(-nu*|x|^2)/...


if 'NSbF' not in locals():
    if rxmimo == 'vae':
        rx["NSymbFrame"]    = int(2e4)
    else:
        rx["NSymbFrame"]    = int(1e4)
else:
    rx["NSymbFrame"]        = int(NSbF)

# even for CMA as the number of pilots is based on the NSbB
if 'NSbB' not in locals():
    rx["NSymbBatch"]    = int(250)
else:
    rx['NSymbBatch']    = NSbB

tx['pilots_info']       = [['cpr','rand',"same","same","4QAM",10,0]]

win_width               = 10
tx['pn_filt_par']       = {
        'type'          : 'moving_average',
        'ma_type'       : 'gaussian',                                          # {uniform, gaussian}
        'window_size'   : win_width,
        'std_dev'       : win_width/6,                                         # only for gaussian
        'err_tolerance' : 5e-1,
        'niter_max'     : int(1e2),
        'adaptative'    : 0                                                    # {1 == yes, 0 ==  no}
}

if get_time:
    tx['get_time']          = 1
    tx['get_exec_time']     = ['frame',[]]
else:
    tx['get_time']          = 0



###############################################################################
################################## RECEIVER ###################################
###############################################################################


if CFO_or_dnu.lower() != 'none':
    tx['flag_phase']  = 1
else:
    tx['flag_phase']  = 0

rx["mimo"]                  = rxmimo                                                # {cma,vae}

if 'rxmode' not in locals():
    rx['mode']          = "blind"                                             # {blind, pilots}
else:
    rx['mode']          = rxmode 

if rxmimo == "vae":
    rx['mode']          = "blind"                                             # {blind, pilots}

if 'NframesTrain' in locals():
    rx["FrameChannel"]  = NframesTrain


###############################################################################
################################## CHANNEL ####################################
###############################################################################

# -----------------------------------------------------------------------------
# if linear variations -------- [[theta_start],[theta_end],[slope]]

if SoPlin_or_fpol.lower() == 'fpol' or SoPlin_or_fpol.lower() == 'vsop':
    fibre['ThetasLaw']["kind"]      = 'Rwalk'
    fibre["ThetasLaw"]['law']       = 'gauss'
    fibre["ThetasLaw"]['theta_in']  = 0*pi/180                                  # [rad]

    Fpols                           = rx['NSymbFrame']*Vsop**2/(2*pi*tx['Rs'])      # [Hz]
    Std_Pol                         = sqrt((2*pi)**2*Fpols/(2*pi*tx['Rs']))               # [rad²]

    Nf_Th                           = list((pi/2*(tx['Rs']*Th_End/Vsop/rx['NSymbFrame'])**2).astype(int))
    paramPOL                        = [list(Vsop),list(Fpols),list(Std_Pol)]

elif SoPlin_or_fpol.lower() == 'soplin':
    fibre['ThetasLaw']["kind"]      = 'func'
    fibre['ThetasLaw']["law"]       = 'lin'
    fibre["ThetasLaw"]['Start']     = 0
    fibre["ThetasLaw"]['End']       = Th_End
    fibre["ThetasLaw"]['SoPlin']    = SoPlin
    Nf_Th                           = ((Th_End*tx['Rs']/(2*pi*SoPlin*rx['NSymbFrame'])).astype(int))
    paramPOL                        = [[SoPlin]]

    # keep the double list format

else:
    fibre['ThetasLaw']["kind"]      = 'func'
    fibre['ThetasLaw']["law"]       = 'lin'
    fibre["ThetasLaw"]['Start']     = 0
    fibre["ThetasLaw"]['End']       = 0
    fibre["ThetasLaw"]['SoPlin']    = 0
    Nf_Th                           = 0
    paramPOL                        = [[0]]

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

if CFO_or_dnu.lower() == 'cfo':
    rx['CFO']               = 1
    tx['PhiLaw']["kind"]    = 'func'
    tx['PhiLaw']["law"]     = 'lin'
    tx["PhiLaw"]['Start']   = 0*pi/180                                                  # [rad]

    paramPHI                = list(CFO)
    Nf_Ph                   = list((Ph_End*tx['Rs']/2/pi/rx['NSymbFrame']/CFO).astype(int))

else:
    rx['CFO']               = 0
    tx['PhiLaw']["kind"]    = 'Rwalk'
    tx['PhiLaw']["law"]     = 'linewidth'

    paramPHI                = list(Dnus)
    Nf_Ph                   = list(((Ph_End/2)**2/rx['NSymbFrame']*tx['Rs']/Dnus).astype(int))

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
fibre['PMD']                    = 0                                            # [ps/sqrt(km)]
fibre['D']                      = 17                                           # [ps/nm/km]
fibre['PMD_SI']                 = fibre['PMD']*1e-12/np.sqrt(1e3)             # [s/sqrt(m)], 1e-12/np.sqrt(1e3) = 1 [ps/sqrt(km)]
fibre['D_SI']                   = fibre['D']*1e-6                              # [s/m/m]
wavelength                      = 1550 *1e-9                                   # [m]
c                               = 299792458                                    # [m/s]

nSymbResCD                      = 0
beta2                           = -wavelength**2/2/pi/c*fibre['D_SI']          # [s²/rad/m]
fibre['DistRes_SI']             = nSymbResCD**2/(tx["Rs"]**2)/abs(beta2)/2/pi  # [m]
fibre['DistProp_SI']            = 1e4                                          # [m]

fibre['tauCD']                  = sqrt(2*pi*fibre['DistRes_SI']*abs(beta2))     # differential group delay [s]
fibre['tauPMD']                 = fibre['PMD_SI']*sqrt(fibre['DistProp_SI'])    # PMD induced delay [s]
# -----------------------------------------------------------------------------

# check 'learning rate tuning cma' for curves

if 'lr' not in locals():
    if rx['mimo'].lower() == "cma":
        rx['lr']    = 1e-6
    else:
        rx['lr']    = 750e-6
else:
    rx['lr']        = lr


#%% ===========================================================================
# --- FUNCTIONS ---
# =============================================================================
def process_data(nrea,npol,nphi,nsnr,tx,fibre,rx,*varargin):

    if len(varargin) != 0:
        NFramesChannel = varargin[0]
        
# -----------------------------------------------------------------------------
# check 'learning rate tuning cma' for curves
    if rxmimo == 'vae':
        if 'NframesTrain' not in locals():
            if paramSNR[nsnr]   < 20 and tx['Rs'] > 100e9:
                rx['FrameChannel']  = int(np.ceil(13.12+0.07*rx['NSymbBatch']))

            elif paramSNR[nsnr] < 20 and tx['Rs'] < 100e9:
                rx['FrameChannel']  = int(np.ceil(10.88+0.07*rx['NSymbBatch']))

            elif paramSNR[nsnr] > 20 and tx['Rs'] > 100e9:
                rx['FrameChannel']  = int(np.ceil(-0.12+0.03*rx['NSymbBatch']))

            elif paramSNR[nsnr] > 20 and tx['Rs'] < 100e9:
                rx['FrameChannel']  = int(np.ceil(1.88+0.03*rx['NSymbBatch']))
    else:
        rx['FrameChannel'] = 1
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
    if tx['PhiLaw']["kind"]     == 'func':
        tx["PhiLaw"]['CFO']     = paramPHI[nphi]                               # [Hz]
        tx["PhiLaw"]['End']     = Ph_End                                       # [rad]
        
    elif tx['PhiLaw']["kind"]   == 'Rwalk':
        tx["dnu"]               = paramPHI[nphi]                               # [Hz]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
    if SoPlin_or_fpol.lower() == 'fpol' or SoPlin_or_fpol.lower() == 'vsop':
        fibre['vsop']                   = paramPOL[0][npol]
        fibre["ThetasLaw"]['fpol']      = paramPOL[1][npol]
        fibre["ThetasLaw"]['theta_std'] = paramPOL[2][npol]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
    rx['SNRdB']                         = paramSNR[nsnr]
    tx, fibre, rx                       = set_Nsymbols(tx,fibre,rx)

    if 'NFramesChannel' not in locals():
        if 'th' in Nf_lim.lower():
            rx['NFrames'] = rx["FrameChannel"] + Nf_Th[npol]
        else:
            rx['NFrames'] = rx["FrameChannel"] + Nf_Ph[nphi]
    else:
        rx['NFrames'] = rx['FrameChannel'] + NFramesChannel

    tx, rx                              = set_Batches_Frames(tx,rx)
# -----------------------------------------------------------------------------

    if not server:
        if SoPlin_or_fpol.lower() == 'fpol' or SoPlin_or_fpol.lower() == 'vsop':
            ExpectT = sqrt(2/pi)*fibre['vsop']*sqrt(rx['NFramesChannel'])*rx['NSymbFrame']/tx['Rs']
        else:
            ExpectT = 2*pi*(paramPOL[0][npol]/Rs)*rx['NSymbFrame']*rx['NFramesChannel']
    
        if not server:
            if CFO_or_dnu == 'dnu':
                ExpectP = 2*sqrt(rx['NFramesChannel']*rx['NSymbFrame']*tx['dnu']/tx['Rs'])
            else:
                ExpectP = 2*pi*(paramPHI[nphi]/Rs)*rx['NSymbFrame']*rx['NFramesChannel']
    
            print(f"\n Rs       = {Rs*1e-9}")
            print(f" rxmimo   = {rxmimo}")
            print(f" NF       = {rx['NFramesChannel']}")
            print(f" NSbF     = {rx['NSymbFrame']*1e-3}")
            print(f" E-theta  = {ExpectT}")
            print(f" E-phi    = {ExpectP}")
            print(f" VSOP     = {fibre['vsop']*1e-6}" \
                  if SoPlin_or_fpol.lower() == 'fpol' or SoPlin_or_fpol.lower() == 'vsop'
                  else f" SoPlin   = {paramPOL[0][npol]}")
            print(f" DNU      = {tx['dnu']*1e-6}\n" \
                  if CFO_or_dnu.lower() == 'dnu'
                  else f" DNU      = {paramPHI[nphi]}\n")


# -----------------------------------------------------------------------------
    saving["filename"]  = misc.create_xml_file(tx,fibre,rx,saving,gen_xml,nrea)
    seed_id             = (nrea+randint(0,10))*(npol+randint(0,10))*(nsnr+randint(0,10))

    if server:
        try:
            tx,fibre,rx  = processing(tx,fibre,rx,saving,seed_id)
            misc.save2mat(tx,fibre,rx,saving)

        except:
            tmp_fn  = 'fail_'+saving['filename'][20:-1]+'.log'
            f       = open(tmp_fn, "a")

            f.write(saving['filename'][24:-1])
            f.close()

    else:
        tx,fibre,rx = processing(tx,fibre,rx,saving,seed_id)
        misc.save2mat(tx,fibre,rx,saving)

        if plot_th and plot_ph:
            plot.y3_axes(saving,"iteration",'Thetas','Phis','SER',['svg'])

        elif plot_th and not plot_ph:
            plot.y2_axes(saving,"iteration",'Thetas','SER',['svg'])

        elif not plot_th and plot_ph:
            plot.y2_axes(saving,"iteration",'Phis','SER',['svg'])

    return tx,fibre,rx

#%% ===========================================================================
# --- LOGISTICS ---
# =============================================================================

for nsnr in range(len(paramSNR)):
    print(f"\t\t\tSNR = {paramSNR[nsnr]}")

    for nrea in range(Nrea):
        print(f"nrea = {nrea}")

        for npol in range(len(paramPOL[0])):
            for nphi in range(len(paramPHI)):

                if 'NFramesChannel' not in locals():
                    tx,fibre,rx     = process_data(nrea,npol,nphi,nsnr,tx,fibre,rx)
                else:
                    tx,fibre,rx     = process_data(nrea,npol,nphi,nsnr,tx,fibre,rx,NFramesChannel)
                    
cuda.empty_cache()



#%%

if sort_files:
    misc.move_files_to_folder(int(Date[2:4]))
    misc.merge_data_folders(saving)
    path = PWD(show=False)+f'/data-{Date[2:]}'
    misc.remove_n_characters_from_filenames(path, 20)
    misc.organise_files(path)

# =============================================================================
# 
# Simsarian OCF     2017    4  krad/s
# Ogaki     OFC     2003    25 deg/min  aerian and burried
# Barcik    FOAN    2019    4  krad     lab, piezo, mechanical
# Charlton  OE      2017    5  Mrad/s   aerial and burried
# Pittala   ICP     2018    8  Mrad/s   lab, thunder
# 
# =============================================================================
