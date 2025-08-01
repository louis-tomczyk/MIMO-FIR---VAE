% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.4
%   Date            : 2024-07-24
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2023-07-06  (1.0.0)
%   2024-07-10  (1.0.1) extract_ground_truth: now already exported from python
%                       phis_gnd reshaped, FIRgnd unused so removed
%                       flexibility and naming normalisation
%   2024-07-11  (1.0.2) phase noise management
%   2024-07-12  (1.0.3) phase noise management --- for rx['mode'] = 'pilots'
%   2024-07-15  (1.1.3) multiple files processing + poincare plotting without FIRgnd
%   2024-07-15  (1.1.4) thetas.gnd updated with right value when theta_in ~= 0
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

function [thetas, phis] = extract_ground_truth(data,caps,thetas,phis)
    
thetas_in   = data.thetas(caps.Frames.Channel+1:end,1)*180/pi;                  % [deg]
thetas_out  = data.thetas(caps.Frames.Channel+1:end,2)*180/pi;                  % [deg]
thetas.gnd  = thetas_in+thetas_out;                                             % [deg]

if caps.phis_est
    if ~strcmpi(caps.rx_mode,'pilots')
        phis.gnd.all    = data.Phis_gnd(caps.Frames.Channel+1:end,:)*180/pi;   % [deg]
        tmp_phi         = zeros(numel(phis.gnd.all),1);
        for k = 1:caps.NFrames.Channel
            tmp_phi(1+(k-1)*caps.NBatches.Frame:k*caps.NBatches.Frame) = phis.gnd.all(k,:);
        end
        phis.gnd.all    = tmp_phi;

    else
        % 1:end-1 as we removed first and last batch in python processing
        tmp_phi         = data.Phis_gnd(:,2:end-1)*180/pi;   % [deg]
        phis.gnd.all    = zeros(numel(tmp_phi),1);
        for k = 1:caps.NFrames.all
            phis.gnd.all(1+(k-1)*caps.NBatches.FrameCut:k*caps.NBatches.FrameCut) = tmp_phi(k,:);
        end
        phis.gnd.all    = repmat(phis.gnd.all,[1,3]);
    end
else
    phis = NaN;
end

if caps.phis_est
    if strcmpi(caps.rx_mode, 'pilots')
        phis.gnd.channel    = phis.gnd.all(caps.NBatches.Training+1:end,1);
    else
        phis.gnd.channel    = phis.gnd.all;
        phis.gnd            = rmfield(phis.gnd,'all');
    end
end

if caps.plot.poincare

    J = zeros(2,caps.NFrames.Channel);
    for k = 1:caps.NFrames.Channel
        m       = k+caps.NFrames.Training;
        THETA   = data.thetas(m,2);
        J(:,k)  = [cos(THETA), -sin(THETA)];
    end
    
    params = {"marker"  ,'square'   ,...
          "size"    , 100        ,...
          "fill"    , ''};
    Jones2Stockes(J,params);
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   check_if_fibre_prop
%   extract_Hgnd
% ---------------------------------------------



function bool = check_if_fibre_prop(h)

    input = h.frame1.h11{1};
    
    if all(input == input(1),'all')
        bool = false;
    else
        bool = true;
    end
%-----------------------------------------------------


function h_gnd = extract_Hgnd(data)

h_gnd = struct();
for k = 1:data.Nframes
    tmp = data.h_channel_liste(k,:,:);
    h_gnd.(sprintf('frame%i',k)).('h11') = tmp(1);
    h_gnd.(sprintf('frame%i',k)).('h12') = tmp(2);
    h_gnd.(sprintf('frame%i',k)).('h21') = tmp(3);
    h_gnd.(sprintf('frame%i',k)).('h22') = tmp(4);
end

clear data.h_channel_liste
bool = check_if_fibre_prop(h_gnd);
if bool == true
    % means that PMD and CD values were not set to 0
    % and are function of the frequency and then the
    % matrix channel elements (h_{ij}) are different
    % samples to samples.
    for row = 1:2
        for col = 1:2
            for frame = 1:data.Nframes
                tmp = h_gnd.(sprintf('frame%i',frame)).(sprintf('h%i%i',row,col));
                h_gnd.(sprintf('frame%i',frame)).(sprintf('h%i%i',row,col)) = sum(tmp{1});
            end
        end
    end

else
    % if no PMD nor CD, then the h_{ij} are the same
    % for all samples, so we need only one element
    for row = 1:2
        for col = 1:2
            for frame = 1:data.Nframes
                tmp = h_gnd.(sprintf('frame%i',frame)).(sprintf('h%i%i',row,col));
                h_gnd.(sprintf('frame%i',frame)).(sprintf('h%i%i',row,col)) = tmp{1}(1);
            end
        end
    end
end
clear tmp

% before it was removed (1.0.0)
% h_gnd       = extract_Hgnd(data);
% h11s        = zeros(data.Nframes-data.FrameChannel,1);
% h12s        = zeros(data.Nframes-data.FrameChannel,1);
% h21s        = zeros(data.Nframes-data.FrameChannel,1);
% h22s        = zeros(data.Nframes-data.FrameChannel,1);
% 
% for k = 1:data.Nframes-data.FrameChannel
%     h11s(k)         = h_gnd.(sprintf('frame%i',k+data.FrameChannel)).h11;
%     h12s(k)         = h_gnd.(sprintf('frame%i',k+data.FrameChannel)).h12;
%     h21s(k)         = h_gnd.(sprintf('frame%i',k+data.FrameChannel)).h21;
%     h22s(k)         = h_gnd.(sprintf('frame%i',k+data.FrameChannel)).h22;
% 
%     ratio_hij_H     = abs(h12s(k)/h11s(k));
%     ratio_hij_V     = abs(h21s(k)/h22s(k));
% 
%     thetas_gnd(1,k) = atan(ratio_hij_H)*180/pi;   % [deg]
%     thetas_gnd(2,k) = atan(ratio_hij_V)*180/pi;   % [deg]
% end
% 
% thetas_gnd(3,:)     = mean(thetas_gnd(1:2,:));  % [deg]
% FIRgnd.HH           = h11s;
% FIRgnd.HV           = h12s;
% FIRgnd.VH           = h21s;
% FIRgnd.VV           = h22s;
%-----------------------------------------------------



