% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Arxivs          :
%   Date            : 2024-07-09
%   Version         : 1.1.0
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-06 (1.0.0)  creation
%   2024-07-09 (1.1.0)  [NEW] plot_SOP, plot_FIR, plot_phi
%                       plot_results: encapsulation + plot phase estimation
%
% ----- MAIN IDEA -----
% ----- INPUTS -----
% ----- OUTPUTS -----
% ----- BIBLIOGRAPHY -----
%   Articles/Books
%   Authors             : [A1]
%   Title               :
%   Jounal/Editor       :
%   Volume - N°         :
%   Date                :
%   DOI/ISBN            :
%   Pages               :
%  ----------------------
%   Codes
%   Author              : [C1] louis tomczyk, Diane Prato
%   Author contact      : louis.tomczyk@telecom-paris.fr
%   Affiliation         : Télécom Paris
%   Date                : 2024-04-19
%   Title of program    : MIMO VAE-FIR
%   Code version        : 1.1.4
%   Web Address         : github.com/louis-tomczyk/MIMO-FIR---VAE
% ---------------------------------------------
%%

function plot_results(caps,H_est, thetas_gnd, thetas_est,phis_est,metrics)

H_ests_norm = prepare_plots(H_est);

if caps.flags.fir

    plot_fir(caps,H_ests_norm);
    plot_SOP(caps,thetas_est,thetas_gnd,metrics);
    f2 = plot_phi(caps,phis_est);
    

    cd ../figs/fir
    exportgraphics(f2,sprintf("%s.png",caps.filename))
    cd(caps.myInitPath)
    pause(0.25)

    if caps.flags.close
        close all
    end
    
end

cd ../err
writematrix(metrics.Err,strcat('Err Theta-',caps.filename,'.csv'))
cd(caps.myInitPath)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   plot_fir            (1.1.0)
%   plot_phi            (1.1.0)
%   plot_SOP            (1.1.0)
%   prepare_plots
% ---------------------------------------------


function H_ests_norm = prepare_plots(H_est)

H_est_11    = squeeze(H_est(1,1,:));
H_est_12    = squeeze(H_est(1,2,:));
H_est_21    = squeeze(H_est(2,1,:));
H_est_22    = squeeze(H_est(2,2,:));
H_ests_abs  = abs([H_est_11,H_est_12,H_est_21,H_est_22]);
norm_factor = max(max(H_ests_abs));
H_ests_norm = H_ests_abs/norm_factor;
% ---------------------------------------------


function f2 = plot_fir(caps,H_ests_norm)

f2 = figure(1);
    subplot(2,2,1);
        hold on
        plot(caps.FIRtaps,abs(H_ests_norm(:,1)),LineWidth=5,Color='k')
        plot(caps.FIRtaps,abs(H_ests_norm(:,4)),'--',color = ones(1,3)*0.83,LineWidth=2)
        xlabel("filter taps")
        ylabel("amplitude")
        legend("$h_{11}$","$h_{22}$")
        axis([-10,10,0,1])

    subplot(2,2,2);
    hold on
        plot(caps.FIRtaps,abs(H_ests_norm(:,2)),LineWidth=5,color = 'k')
        plot(caps.FIRtaps,abs(H_ests_norm(:,3)),'--',color = ones(1,3)*0.83,LineWidth=2)
        xlabel("filter taps")
        legend("$h_{12}$","$h_{21}$")
        axis([-10,10,0,1])
% ---------------------------------------------



function f2 = plot_SOP(caps,thetas_est,thetas_gnd,metrics)

f2 = figure(1);
if caps.flags.plot.phi
    subplot(2,2,3)
else
    subplot(2,2,[3,4])
end

hold on
    if strcmpi(caps.flags.SOP,'error per frame')
        scatter(caps.frames,thetas_est-thetas_gnd(3,:),100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
        xlabel("frame")
        ylabel("$\hat{\theta}-\theta$ [deg]")

    elseif strcmpi(caps.flags.SOP,'error per theta')
        scatter(thetas_gnd(3,:),thetas_est-thetas_gnd(3,:),100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
        xlabel("$\theta$ [deg]")
        ylabel("$\hat{\theta}-\theta$ [deg]")

    elseif strcmpi(caps.flags.SOP,'comparison per frame')
        plot(caps.frames,thetas_gnd(3,:),'color',[1,1,1]*0.83, LineWidth=5)
        scatter(caps.frames,thetas_est,100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
        legend("ground truth","estimation",Location="northwest")
        xlabel("frame")
        ylabel("$\hat{\theta},\theta$ [deg]")
    end

    title(sprintf("%s - tap = %d, Error to ground truth = %.2f +/- %.1f [deg]", ...
        caps.method.thetas, caps.tap,metrics.ErrMean(caps.kdata),metrics.ErrStd(caps.kdata)))
% ---------------------------------------------




function f2 = plot_phi(caps,phis_est)

f2 = figure(1);
if caps.flags.plot.phi
    subplot(2,2,4)

    phi_ground = linspace(0,10,caps.NFramesChannel);
    hold on
    plot(caps.frames,phi_ground,'--',color = ones(1,3)*0.83,LineWidth=2)
    scatter(caps.frames,phis_est(caps.FrameChannel+1:end),100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
    xlabel("frame")
    ylabel("$\hat{\phi}$ [deg]")
    

end
% ---------------------------------------------
