% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Date            : 2024-07-11
%   Version         : 1.1.1
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
% 
% ----- CHANGE LOG -----
%   2024-07-06 (1.0.0)  creation
%   2024-07-09 (1.1.0)  [NEW] check_fir, check_orthogonality
%                       extract_thetas_est: fftshift for natural FIR checking
%                       extract_phis_est: checking orthogonality + working phase estimation
%   2024-07-11 (1.1.1)  managing phase noises
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

function [thetas,phis, H_est, f, Sest,FIRest] = channel_estimation(Dat,caps)

data            = Dat{caps.kdata};
H_est           = zeros(2,2,caps.FIRlength);
thetas.est      = zeros(caps.NFramesChannel-1,1);
H_est_f         = zeros([caps.NFramesChannel,size(H_est)]);

if caps.est_phi == 1
    phis.est    = zeros(caps.NBatchFrame,caps.NFramesChannel);
else
    phis        = NaN;
end

for k = 1:caps.NFramesChannel
    caps.Frame              = caps.FrameChannel+k;
    H_est                   = extract_Hest(data,caps);
    [thetas.est(k),H_est_f] = extract_thetas_est(H_est,k,H_est_f,caps); % [deg]

    if caps.est_phi
        for j = 1:caps.NBatchFrame
            caps.batch      = (caps.Frame-1)*caps.NBatchFrame+j;
            H_est           = extract_Hest(data,caps,H_est);
            phis.est(j,k)   = extract_phis_est(H_est,caps);             % [deg]
        end
    end
end

tmp = phis.est;
phis.est = zeros(caps.NFramesChannel*caps.NBatchFrame,1);
for k = 1:caps.NFramesChannel
    phis.est(1+(k-1)*caps.NBatchFrame:k*caps.NBatchFrame,1) = tmp(:,k);
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
    saveas(f,sprintf("%s --- Poincare.png",caps.filename))
    cd(caps.myInitPath)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   check_fir                       (1.1.0)
%   check_orthogonality             (1.1.0)
%   extract_Hest
%   extract_phis_est
%   extract_thetas_est
% ---------------------------------------------



function check_fir(H_est,caps)

if caps.plot.fir
    H11     = reshape(H_est(1,1,:),1,[]);
    H12     = reshape(H_est(1,2,:),1,[]);
    H21     = reshape(H_est(2,1,:),1,[]);
    H22     = reshape(H_est(2,2,:),1,[]);
    Mreal   = max(abs([real(H11),real(H12),real(H21),real(H22)]));
    Mimag   = max(abs([imag(H11),imag(H12),imag(H21),imag(H22)]));

    t   = linspace(1,caps.FIRlength,caps.FIRlength)-ceil(caps.FIRlength/2);
    figure
    subplot(2,2,1)
        hold on
        plot([-1,1]*caps.FIRlength/2,[0,0],LineWidth= 3, Color='k')
        plot(t,real(H11))
        plot(t,fliplr(real(H22)))
        title('real part')
        ylim([-1,1]*Mreal)
        legend("-","11","22")
    subplot(2,2,2)
        hold on
        plot([-1,1]*caps.FIRlength/2,[0,0],LineWidth= 3, Color='k')
        plot(t,imag(H11))
        plot(t,fliplr(imag(H22)))
        title('imag part')
        ylim([-1,1]*Mimag)
        legend("-","11","22")

    subplot(2,2,3)
        hold on
        plot([-1,1]*caps.FIRlength/2,[0,0],LineWidth= 3, Color='k')
        plot(t,real(H12))
        plot(t,fliplr(real(H21)))
        title('real part')
        ylim([-1,1]*Mreal)
        legend("-","12","21")
    subplot(2,2,4)
        hold on
        plot([-1,1]*caps.FIRlength/2,[0,0],LineWidth= 3, Color='k')
        plot(t,imag(H12))
        plot(t,fliplr(imag(H21)))
        title('imag part')
        ylim([-1,1]*Mimag)
        legend("-","12","21")
end
% ---------------------------------------------



function check_orthogonality(Hest,caps)

if ~isfield(caps,'check')
    caps.check.fir_orth = 0;
end

if caps.check.fir_orth
    H           = Hest(:,:,caps.tap);
    tmp         = sum(sum(H'*H - H*H' < 1e-2*ones(2)))/4;
    M           = H*H';
%     diagMeanR   = trace(real(M))/2;
%     diagMeanR   = trace(imag(M))/2;
%     fprintf("%i,%d,%d,%.2f,%.2f\n", k,caps.tap,tmp == 1, diagMeanR, diagMeanI)
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
    h_est_11_I_j        = reshape(data.h_est_frame(caps.Frame,1,1,1,:),[1,caps.FIRlength]);
    h_est_12_I_j        = reshape(data.h_est_frame(caps.Frame,1,2,1,:),[1,caps.FIRlength]);
    h_est_21_I_j        = reshape(data.h_est_frame(caps.Frame,2,1,1,:),[1,caps.FIRlength]);
    h_est_22_I_j        = reshape(data.h_est_frame(caps.Frame,2,2,1,:),[1,caps.FIRlength]);
    
    h_est_11_Q_j        = reshape(data.h_est_frame(caps.Frame,1,1,2,:),[1,caps.FIRlength]);
    h_est_12_Q_j        = reshape(data.h_est_frame(caps.Frame,1,2,2,:),[1,caps.FIRlength]);
    h_est_21_Q_j        = reshape(data.h_est_frame(caps.Frame,2,1,2,:),[1,caps.FIRlength]);
    h_est_22_Q_j        = reshape(data.h_est_frame(caps.Frame,2,2,2,:),[1,caps.FIRlength]);

    H_est.frame(1,1,:)  = complex(h_est_11_I_j, h_est_11_Q_j);
    H_est.frame(1,2,:)  = complex(h_est_12_I_j, h_est_12_Q_j);
    H_est.frame(2,1,:)  = complex(h_est_21_I_j, h_est_21_Q_j);
    H_est.frame(2,2,:)  = complex(h_est_22_I_j, h_est_22_Q_j);

    check_orthogonality(H_est.frame,caps)
else

    h_est_11_I_j        = reshape(data.h_est_batch(caps.batch,1,1,1,:),[1,caps.FIRlength]);
    h_est_12_I_j        = reshape(data.h_est_batch(caps.batch,1,2,1,:),[1,caps.FIRlength]);
    h_est_21_I_j        = reshape(data.h_est_batch(caps.batch,2,1,1,:),[1,caps.FIRlength]);
    h_est_22_I_j        = reshape(data.h_est_batch(caps.batch,2,2,1,:),[1,caps.FIRlength]);
    
    h_est_11_Q_j        = reshape(data.h_est_batch(caps.batch,1,1,2,:),[1,caps.FIRlength]);
    h_est_12_Q_j        = reshape(data.h_est_batch(caps.batch,1,2,2,:),[1,caps.FIRlength]);
    h_est_21_Q_j        = reshape(data.h_est_batch(caps.batch,2,1,2,:),[1,caps.FIRlength]);
    h_est_22_Q_j        = reshape(data.h_est_batch(caps.batch,2,2,2,:),[1,caps.FIRlength]);

    H_est.batch(1,1,:)  = complex(h_est_11_I_j, h_est_11_Q_j);
    H_est.batch(1,2,:)  = complex(h_est_12_I_j, h_est_12_Q_j);
    H_est.batch(2,1,:)  = complex(h_est_21_I_j, h_est_21_Q_j);
    H_est.batch(2,2,:)  = complex(h_est_22_I_j, h_est_22_Q_j);

    check_orthogonality(H_est.batch,caps)
end


% ---------------------------------------------

function [phis_est, g0] = extract_phis_est(H_est,caps)

if caps.Frame > caps.FrameChannel
    Hest        = H_est.batch(:,:,caps.tap);
    M           = Hest'*Hest;
    g0          = mean(trace(M));
    phis_est    = 0.5*angle(det(Hest))*180/pi;

else
    phis_est    = 0;

end

% ---------------------------------------------

function [thetas_est, Hest_f] = extract_thetas_est(H_est,k,Hest_f,caps)

Hest_f(k,1,1,:) = fftshift(fft(H_est.frame(1,1,:)));
Hest_f(k,1,2,:) = fftshift(fft(H_est.frame(1,2,:)));
Hest_f(k,2,1,:) = fftshift(fft(H_est.frame(2,1,:)));
Hest_f(k,2,2,:) = fftshift(fft(H_est.frame(2,2,:)));

Hest            = H_est.frame(:,:,caps.tap);

if strcmpi(caps.method.thetas,'fft')
    H_f0        = squeeze(Hest_f(k,:,:,caps.tap));
    thetas_est  = atan(abs(H_f0(1,2)./H_f0(1,1)))*180/pi;   % [deg]

elseif strcmpi(caps.method.thetas,'svd')
    [U,~,~]     = svd(Hest);
    Cos11       = -U(1,1);
    Cos22       = U(2,2);
    Sin12       = U(1,2);
    Sin21       = U(2,1);
    
    Tcos11      = acos(Cos11)*180/pi;
    Tcos22      = acos(Cos22)*180/pi;
    Tsin12      = asin(Sin12)*180/pi;
    Tsin21      = asin(Sin21)*180/pi;
    thetas_est  = real(Tcos11);

elseif strcmpi(caps.method.thetas,'mat')
    Tr          = sum(eig(Hest));
    D           = prod(eig(Hest));
    C           = Tr/2/sqrt(D);
    thetas_est  = real(acos(C)*180/pi);
end
% ---------------------------------------------

