% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Date            : 2024-07-10
%   Version         : 1.1.1
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-06 (1.0.0)  creation
%   2024-07-09 (1.1.0)  [NEW] plot_SOP, plot_FIR, plot_phi
%                       plot_results: encapsulation + plot phase estimation
%   2024-07-10 (1.1.1)  flexibility and naming standardisation
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

function plot_results(caps,H_est,thetas,metrics,varargin)

if ~isempty(varargin)
    phis.gnd    = varargin{1}.gnd;
    phis.est    = varargin{1}.est;
end


H_ests_norm = prepare_plots(H_est);

if caps.flags.fir

    plot_fir(caps,H_ests_norm);
    f = plot_SOP(caps,thetas,metrics);

    if ~isempty(varargin)
        f = plot_phi(caps,phis);
    end

    cd ../figs/fir
    exportgraphics(f,sprintf("%s.png",caps.filename))
    cd(caps.myInitPath)
    pause(0.25)

    if caps.flags.close
        close all
    end
    
end

cd ../err
writematrix(metrics.Err.thetas,strcat('Err Theta-',caps.filename,'.csv'))

if ~isempty(varargin)
    writematrix(metrics.Err.phis,strcat('Err Phi-',caps.filename,'.csv'))
end

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


function f = plot_fir(caps,H_ests_norm)

f = figure(1);
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



function f = plot_SOP(caps,thetas,metrics)

f = figure(1);
if caps.flags.plot.phi
    subplot(2,2,3)
else
    subplot(2,2,[3,4])
end

hold on
if strcmpi(caps.flags.SOP,'error per frame')
    scatter(caps.Frames,thetas.est-thetas.gnd,100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
    xlabel("frame")
    ylabel("$\hat{\theta}-\theta$ [deg]")

elseif strcmpi(caps.flags.SOP,'error per theta')
    scatter(thetas.gnd,thetas.est-thetas.gnd,100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
    xlabel("$\theta$ [deg]")
    ylabel("$\hat{\theta}-\theta$ [deg]")

elseif strcmpi(caps.flags.SOP,'comparison per frame')
    plot(caps.Frames,thetas.gnd,'color',[1,1,1]*0.83, LineWidth=5)
    scatter(caps.Frames,thetas.est,100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
    legend("ground truth","estimation",Location="northwest")
    xlabel("frame")
    ylabel("$\hat{\theta},\theta$ [deg]")
end

title(sprintf("%s - tap = %d, Error to ground truth = %.2f +/- %.1f [deg]", ...
      caps.method.thetas, caps.tap, ...
      metrics.thetas.ErrMean(caps.kdata),metrics.thetas.ErrStd(caps.kdata)))
% ---------------------------------------------




function f = plot_phi(caps,phis)

f = figure(1);
if caps.flags.plot.phi
    subplot(2,2,4)

    hold on
    plot(caps.Frames,phis.gnd,'--',color = ones(1,3)*0.83,LineWidth=2)
    scatter(caps.Frames,phis.est(caps.FrameChannel+1:end),100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
    xlabel("frame")
    ylabel("$\hat{\phi}$ [deg]")
    

end
% ---------------------------------------------
