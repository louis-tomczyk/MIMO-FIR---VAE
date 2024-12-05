% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.0.0
%   Date            : 2024-12.01
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
%   2024-07-11 (1.1.2)  plot for phase estimation per batch (1.1.0 == per frame)
%   2024-07-12 (1.1.3)  phase noise management --- for rx['mode'] = 'pilots'
%   2024-07-16 (1.1.4)  multiple files processing
%   2024-07-26 (1.1.5)  lighten the filenames + include estimation method
%   2024-10-28 (1.2.0)  *plot_fir/plot_sop* (1.1.1) mean -> median
%   2024-12-01 (2.0.0)  saving independently the plots
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


if caps.save.data

    if ~isempty(varargin)
        data.phis           = phis;
    end
    data.thetas         = thetas;
    data.H_ests_norm    = H_ests_norm;
    
    cd(caps.log.myRootPath)
    if ~isfolder("saved_data")
        mkdir saved_data
    end
    cd saved_data/
    caps.log.mySavePath = pwd();

    caps.log.fname_save = strcat("saved_",caps.log.filename(7:end));
    
    save(strcat(caps.log.fname_save,'.mat'),"data")
    cd(caps.log.myInitPath)

end


if caps.plot.fir

    plot_fir(caps,H_ests_norm,caps.save.data);
    f = plot_SOP(caps,thetas,metrics);

    if ~isempty(varargin)
        f = plot_phi(caps,phis,metrics);
    end

    if ~caps.save.data
        cd ../figs/fir
        exportgraphics(f,sprintf("%s.png",caps.log.filename))
        cd(caps.log.myInitPath)
        pause(0.25)
    end

    if caps.plot.close || caps.save.data
        close all
    end


end

if caps.save.errs
    cd ../err/thetas/
    str_tmp         = strcat( ...
                        sprintf('ErrTh-%s-',caps.method.thetas), ...
                        caps.log.filename(7:end),'.csv'); % 7 == 'X - mimo...'
    
    writematrix(metrics.thetas.Err,str_tmp)
    
    if ~isempty(varargin)
        cd ../phis/
        str_tmp     = strcat( ...
                    sprintf('ErrPh-%s-',caps.method.phis), ...
                    caps.log.filename(7:end),'.csv'); % 7 == 'X - mimo...'
        writematrix(metrics.phis.Err,str_tmp)
    end
end


cd(caps.log.myInitPath)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   plot_fir            (1.2.0)
%   plot_phi            (1.2.0)
%   plot_SOP            (1.2.0)
%   prepare_plots
% ---------------------------------------------


function H_ests_norm = prepare_plots(H_est)

if isfield(H_est,'frame')
    H_est   = H_est.frame;
else
    H_est   = H_est.batch;
end

H_est_11    = squeeze(H_est(1,1,:));
H_est_12    = squeeze(H_est(1,2,:));
H_est_21    = squeeze(H_est(2,1,:));
H_est_22    = squeeze(H_est(2,2,:));
H_ests_abs  = abs([H_est_11,H_est_12,H_est_21,H_est_22]);
norm_factor = max(max(H_ests_abs));
H_ests_norm = H_ests_abs/norm_factor;
% ---------------------------------------------


function f = plot_fir(caps,H_ests_norm,varargin)

f = figure(1);
    if isempty(varargin) || varargin{1}==0
        subplot(2,2,1);
    end

    hold on
    plot(caps.FIR.taps,abs(H_ests_norm(:,1)),LineWidth=5,Color='k')
    plot(caps.FIR.taps,abs(H_ests_norm(:,4)),'--',color = ones(1,3)*0.83,LineWidth=2)
    xlabel("filter taps")
    ylabel("amplitude")
    legend("$h_{11}$","$h_{22}$")
    axis([-caps.FIR.length/2-1,caps.FIR.length/2+1,0,1])


    if isempty(varargin) || varargin{1}==0
        subplot(2,2,2);
        hold on
            plot(caps.FIR.taps,abs(H_ests_norm(:,2)),LineWidth=5,color = 'k')
            plot(caps.FIR.taps,abs(H_ests_norm(:,3)),'--',color = ones(1,3)*0.83,LineWidth=2)
            xlabel("filter taps")
            legend("$h_{12}$","$h_{21}$")
            axis([-caps.FIR.length/2-1,caps.FIR.length/2+1,0,1])
    end

    if ~isempty(varargin) && varargin{1}==1
        cd(caps.log.mySavePath)
        set(f,"Position",[0.0198,0.0009,0.5255,0.8824]);
        saveas(f,strcat('FIR_',caps.log.fname_save,'.fig'))
        saveas(f,strcat('FIR_',caps.log.fname_save,'.png'))
        cd(caps.log.myInitPath)
        close all
    end

% ---------------------------------------------



function f = plot_SOP(caps,thetas,metrics)

if ~caps.save.data
    f = figure(1);
    if caps.phis_est
        subplot(2,2,3)
    else
        subplot(2,2,[3,4])
    end
else
     f = figure("Position",[0.0198,0.0009,0.5255,0.8824]);
end

hold on
if strcmpi(caps.plot.SOP.xlabel,'error per frame')
    scatter(caps.Frames.array,thetas.est-thetas.gnd,100,"filled", ...
        MarkerEdgeColor="k",MarkerFaceColor='k')
    xlabel("frame")
    ylabel("$\hat{\theta}-\theta$ [deg]")

elseif strcmpi(caps.plot.SOP.xlabel,'error per theta')
    scatter(thetas.gnd,thetas.est-thetas.gnd,100,"filled", ...
        MarkerEdgeColor="k",MarkerFaceColor='k')
    xlabel("$\theta$ [deg]")
    ylabel("$\hat{\theta}-\theta$ [deg]")

elseif strcmpi(caps.plot.SOP.xlabel,'comparison per frame')
    scatter(caps.Frames.array,thetas.est,100,"filled", ...
        MarkerEdgeColor="k",MarkerFaceColor='k', ...
        DisplayName="estimation")
    plot(caps.Frames.array,thetas.gnd,'color',[1,1,1]*0.83, LineWidth=5, ...
        DisplayName="ground truth")

    legend(Location="northwest")
    xlabel("frame")
    ylabel("$\hat{\theta},\theta$ [deg]")
end

title(sprintf("%s - tap = %d, Error to ground truth = %.2f +/- %.1f [deg]", ...
      caps.method.thetas, caps.FIR.tap, ...
      metrics.thetas.ErrMedian,metrics.thetas.ErrStd))

if caps.save.data
    cd(caps.log.mySavePath)
    saveas(f,strcat('SoP_',caps.log.fname_save,'.fig'))
    saveas(f,strcat('SoP_',caps.log.fname_save,'.png'))
    cd(caps.log.myInitPath)
    f = clf();
end


% ---------------------------------------------




function f = plot_phi(caps,phis,metrics)

f = figure(1);
if caps.phis_est
    if ~caps.save.data
        subplot(2,2,4)
    end

    hold on
    if strcmpi(caps.plot.phis.xlabel,'error per batch')
        scatter(caps.Batches.array,...
                phis.est.channel(:,caps.plot.phis.pol)-phis.gnd.channel(:,caps.plot.phis.pol),...
                25,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
        xlabel("batch")
        ylabel("$\hat{\phi}-\phi$ [deg]")
    
    elseif strcmpi(caps.plot.phis.xlabel,'error per phi')
        scatter(phis.gnd.channel(:,caps.plot.phis.pol), ...
                phis.est.channel(:,caps.plot.phis.pol)-phis.gnd.channel(:,caps.plot.phis.pol), ...
                25,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
        xlabel("$\phi$ [deg]")
        ylabel("$\hat{\phi}-\phi$ [deg]")
    
    elseif strcmpi(caps.plot.phis.xlabel,'comparison per batch')
        scatter(caps.Batches.array,phis.est.channel(:,caps.plot.phis.pol),...
                25,"filled",MarkerEdgeColor="k",MarkerFaceColor='k', ...
                DisplayName="estimation")
        plot(caps.Batches.array, ...
             phis.gnd.channel, ...
             'color',[1,1,1]*0.83, LineWidth=2, ...
             DisplayName="ground truth")

        if ~isempty(find(phis.gnd.channel>90, 1))
            plot(caps.Batches.array,ones(caps.NBatches.Channel,1)*90,...
                '-.k', LineWidth=1, ...
                HandleVisibility="off")
        elseif ~isempty(find(phis.gnd.channel<-90, 1))
            plot(caps.Batches.array,-ones(caps.NBatches.Channel,1)*90,...
                '-.k', LineWidth=1, ...
                HandleVisibility="off")
        end

        legend(Location="northwest")
        xlabel("batch")
        ylabel("$\hat{\phi},\phi$ [deg]")
    end

title(sprintf("%s - tap = %d, Error to ground truth = %.2f +/- %.1f [deg]", ...
      caps.method.phis, caps.FIR.tap, ...
      metrics.phis.ErrMedian(caps.kdata),metrics.phis.ErrStd(caps.kdata)))
end

if caps.save.data
    cd(caps.log.mySavePath)
    saveas(f,strcat('Phi_',caps.log.fname_save,'.fig'))
    saveas(f,strcat('Phi_',caps.log.fname_save,'.png'))
    cd(caps.log.myInitPath)
    f = clf();
end

% ---------------------------------------------
