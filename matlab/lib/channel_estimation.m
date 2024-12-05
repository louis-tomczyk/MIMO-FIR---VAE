% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.0.0
%   Date            : 2024-11-13
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
% 
% ----- CHANGE LOG -----
%   2024-07-06  (1.0.0) creation
%   2024-07-09  (1.1.0) [NEW] check_fir, check_orthogonality
%                       extract_thetas_est: fftshift for natural FIR checking
%                       extract_phis_est: checking orthogonality + working phase estimation
%   2024-07-11  (1.1.1) phase noise management
%   2024-07-12  (1.1.2) phase noise management --- for rx['mode'] = 'pilots'
%                       [REMOVED] check_fir
%   2024-07-15  (1.1.3) multiple files processing
%   2024-09-26  (1.2.0) extract_thetas_est (1.1.0): removing abs
%                       to measure negative angles + cleaning
% ----------------------
%   2024-11-13  (2.0.0) [NEW] HANDMADE_UNWRAP for the phase
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

function [caps, thetas, phis, H_est, f, Sest, FIRest] = channel_estimation(data,caps)

H_est           = zeros([2,2,caps.FIR.length]);
thetas.est      = zeros([caps.NFrames.Channel-1,1]);
H_est_f         = zeros([caps.NFrames.Channel,size(H_est)]);

if caps.phis_est == 1 && ~strcmpi(caps.rx_mode,'pilots')
    phis.est.all    = zeros([caps.NBatches.Frame,caps.NFrames.Channel]);

elseif caps.phis_est == 1 && strcmpi(caps.rx_mode,'pilots')
    % 3 = polH, polV, mean(polH,polV)
    phis.est.all    = zeros([caps.NBatches.FrameCut*caps.NFrames.all,3]);
else
    phis        = NaN;
end

for k = 1:caps.NFrames.Channel
    caps.Frame              = caps.Frames.Channel+k;
    H_est                   = extract_Hest(data,caps);
    [thetas.est(k),H_est_f] = extract_thetas_est(H_est,k,H_est_f,caps); % [deg]

    if ~strcmpi(caps.rx_mode,'pilots') &&  caps.phis_est
        for j = 1:caps.NBatches.Frame
            caps.batch          = (caps.Frame-1)*caps.NBatches.Frame+j;
            H_est               = extract_Hest(data,caps,H_est);
            phis.est.all(j,k)   = extract_phis_est(H_est,caps);             % [deg]
        end
    end
end
if caps.unwrap
    thetas.est    = handmade_unwrap(thetas.est);
end

% handmade flattening
if caps.phis_est
    if ~strcmpi(caps.rx_mode, 'pilots') && ~isempty(phis.est) % if vae
        tmp             = phis.est.all;
        phis.est.all    = zeros(caps.NFrames.Channel*caps.NBatches.Frame,1);
        for k = 1:caps.NFrames.Channel
            phis.est.all(1+(k-1)*caps.NBatches.Frame:k*caps.NBatches.Frame,1) = tmp(:,k);
        end
        if caps.unwrap
            phis.est.all    = handmade_unwrap(phis.est.all);
        end

    elseif strcmpi(caps.rx_mode, 'pilots')
        polH = squeeze(data.PhaseNoise_est_cpr(:,1,:));
        polV = squeeze(data.PhaseNoise_est_cpr(:,2,:));
    
        for k = 1:caps.NFrames.all
            phis.est.all(1+(k-1)*caps.NBatches.FrameCut:k*caps.NBatches.FrameCut,1) = polH(k,:)';
            phis.est.all(1+(k-1)*caps.NBatches.FrameCut:k*caps.NBatches.FrameCut,2) = polV(k,:)';
        end
    
        phis.est.all(:,3) = mean(phis.est.all(:,1:2), 2);
    else
    end

    if caps.phis_est && strcmpi(caps.rx_mode, 'pilots')
        phis.est.channel = phis.est.all(caps.NBatches.Training+1:end,:);
        
        if ~isfield(caps.plot.phis, 'pol')
            caps.plot.phis.pol = 3;          % 1 = polH, 2 = polV, 3 = mean
        end
    else
        phis.est.channel    = phis.est.all;
        phis.est            = rmfield(phis.est,'all');
    end
end

% to show the FIR filter of the last step

FIRest.HH = squeeze(H_est_f(data.FrameChannel+1:end,1,1,:));
FIRest.HV = squeeze(H_est_f(data.FrameChannel+1:end,1,2,:));
FIRest.VH = squeeze(H_est_f(data.FrameChannel+1:end,2,1,:));
FIRest.VV = squeeze(H_est_f(data.FrameChannel+1:end,2,2,:));


if caps.plot.poincare
    params = {"marker"  ,'o'        ,...
              "size"    , 50        ,...
              "fill"    , 'filled'};
    [Sest,f]   = FIR2Stockes(FIRest,params);

    cd ../figs/poincare
    saveas(f,sprintf("%s --- Poincare.png",caps.log.filename))
    cd(caps.log.myInitPath)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   check_orthogonality             (1.1.0)
%   extract_Hest
%   extract_phis_est
%   extract_thetas_est              (1.1.0)
%   handmade_unwrap                 (2.0.0)
% ---------------------------------------------


function check_orthogonality(Hest,caps)

if ~isfield(caps,'check')
    caps.check.fir_orth = 0;
end

if caps.check.fir_orth
    H           = Hest(:,:,caps.FIR.tap);
    tmp         = sum(sum(H'*H - H*H' < 1e-2*ones(2)))/4;
%     M           = H*H';
%     diagMeanR   = trace(real(M))/2;
%     diagMeanR   = trace(imag(M))/2;
%     fprintf("%i,%d,%d,%.2f,%.2f\n", k,caps.FIR.tap,tmp == 1, diagMeanR, diagMeanI)
    assert(tmp == 1,"Hest Hest^dagger not unitary")
end
% ---------------------------------------------

function H_est = extract_Hest(data,caps,varargin)

if ~isempty(varargin)
    what = 'batch';
else
    what = "frame";
end

if strcmpi(what,'frame')
    h_est_11_I_j        = reshape(data.h_est_frame(caps.Frame,1,1,1,:),[1,caps.FIR.length]);
    h_est_12_I_j        = reshape(data.h_est_frame(caps.Frame,1,2,1,:),[1,caps.FIR.length]);
    h_est_21_I_j        = reshape(data.h_est_frame(caps.Frame,2,1,1,:),[1,caps.FIR.length]);
    h_est_22_I_j        = reshape(data.h_est_frame(caps.Frame,2,2,1,:),[1,caps.FIR.length]);
    
    h_est_11_Q_j        = reshape(data.h_est_frame(caps.Frame,1,1,2,:),[1,caps.FIR.length]);
    h_est_12_Q_j        = reshape(data.h_est_frame(caps.Frame,1,2,2,:),[1,caps.FIR.length]);
    h_est_21_Q_j        = reshape(data.h_est_frame(caps.Frame,2,1,2,:),[1,caps.FIR.length]);
    h_est_22_Q_j        = reshape(data.h_est_frame(caps.Frame,2,2,2,:),[1,caps.FIR.length]);

    H_est.frame(1,1,:)  = complex(h_est_11_I_j, h_est_11_Q_j);
    H_est.frame(1,2,:)  = complex(h_est_12_I_j, h_est_12_Q_j);
    H_est.frame(2,1,:)  = complex(h_est_21_I_j, h_est_21_Q_j);
    H_est.frame(2,2,:)  = complex(h_est_22_I_j, h_est_22_Q_j);

    check_orthogonality(H_est.frame,caps)
else

    if ~isempty(data.h_est_batch)
        h_est_11_I_j        = reshape(data.h_est_batch(caps.batch,1,1,1,:),[1,caps.FIR.length]);
        h_est_12_I_j        = reshape(data.h_est_batch(caps.batch,1,2,1,:),[1,caps.FIR.length]);
        h_est_21_I_j        = reshape(data.h_est_batch(caps.batch,2,1,1,:),[1,caps.FIR.length]);
        h_est_22_I_j        = reshape(data.h_est_batch(caps.batch,2,2,1,:),[1,caps.FIR.length]);
        
        h_est_11_Q_j        = reshape(data.h_est_batch(caps.batch,1,1,2,:),[1,caps.FIR.length]);
        h_est_12_Q_j        = reshape(data.h_est_batch(caps.batch,1,2,2,:),[1,caps.FIR.length]);
        h_est_21_Q_j        = reshape(data.h_est_batch(caps.batch,2,1,2,:),[1,caps.FIR.length]);
        h_est_22_Q_j        = reshape(data.h_est_batch(caps.batch,2,2,2,:),[1,caps.FIR.length]);
    
        H_est.batch(1,1,:)  = complex(h_est_11_I_j, h_est_11_Q_j);
        H_est.batch(1,2,:)  = complex(h_est_12_I_j, h_est_12_Q_j);
        H_est.batch(2,1,:)  = complex(h_est_21_I_j, h_est_21_Q_j);
        H_est.batch(2,2,:)  = complex(h_est_22_I_j, h_est_22_Q_j);
    
        check_orthogonality(H_est.batch,caps)
    end
end


% ---------------------------------------------

function [phis_est, g0] = extract_phis_est(H_est,caps)

if caps.Frame > caps.Frames.Channel
    Hest        = H_est.batch(:,:,caps.FIR.tap);
    M           = Hest'*Hest;
    g0          = mean(trace(M));

    phis_est    = 0.5*angle(det(Hest));
    phis_est    = phis_est*180/pi;
else
    phis_est    = 0;

end

% ---------------------------------------------

function y = handmade_unwrap(x)

    ref         = 90;
    xdiff       = diff(x);
    index_jumps = find(abs(xdiff)>ref);
    njumps      = length(index_jumps);

    count       = 0;
    y           = x;

    while count < njumps
        count       = count+1;
        k           = index_jumps(count);
        offset      = -2*ref*sign(x(k+1)-x(k));
        if count == 1
            y(k+1:end)  = x(index_jumps(count)+1:end)+offset;
        else
            y(k+1:end)  = y(index_jumps(count)+1:end)+offset;
        end
    end

% ---------------------------------------------


function [thetas_est, Hest_f] = extract_thetas_est(H_est,k,Hest_f,caps)

if strcmpi(caps.method.thetas,'fft')
    Hest_f(k,1,1,:) = fftshift(fft(H_est.frame(1,1,:)));
    Hest_f(k,1,2,:) = fftshift(fft(H_est.frame(1,2,:)));
    Hest_f(k,2,1,:) = fftshift(fft(H_est.frame(2,1,:)));
    Hest_f(k,2,2,:) = fftshift(fft(H_est.frame(2,2,:)));

    H_f0        = squeeze(Hest_f(k,:,:,caps.FIR.tap));
    thetas_est  = unwrap(real(atan(H_f0(1,2)./H_f0(1,1))))*180/pi;   % [deg]


elseif strcmpi(caps.method.thetas,'mat')
    Hest        = H_est.frame(:,:,caps.FIR.tap);

%     Tr          = sum(eig(Hest));
%     D           = prod(eig(Hest));
%     C           = Tr/2/sqrt(D);
%     thetas_est  = real(acos(C)*180/pi);

    tanTheta    = Hest(1,2)/Hest(1,1);
    thetas_est  = real(atan(tanTheta)*180/pi);
end
% ---------------------------------------------

