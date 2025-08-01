% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.3
%   Date            : 2024-11-06
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-07 (1.0.0)
%   2024-07-10 (1.1.0)  [NEW] moving_stat: merged version
%                       of 'moving_average/std' 
%                       encapsulating in structures metrics
%                       naming standardisation
%   2024-07-15  (1.1.1) multiple files processing
%   2024-10-28  (1.1.2) mean -> median
%   2024-11-05  (1.1.3) moving average <=> period < length
%                       checking entrance size of thetas
% 
% ----- MAIN IDEA -----
%   Evaluate estimation errors
%
% ----- INPUTS -----
% ----- BIBLIOGRAPHY -----
%   Functions           : main_0_python2matlab
%   Author              : louis tomczyk
%   Author contact      : louis.tomczyk@telecom-paris.fr
%   Date                : 2024-06-26
%   Title of program    : VAE-FIR
%   Code version        : 1.0
%   Type                : source code
%   Web Address         : gitlab.telecom-paris.fr/elie.awwad/vae-fir/-/tree/main/codes/matlab?ref_type=heads
%
%   Functions           : moving_average
%   Author              : louis tomczyk
%   Author contact      : louis.tomczyk@telecom-paris.fr
%   Date                : 2022-05-23
%   Title of program    : VAE-FIR
%   Code version        : 1.0.0
%   Type                : source code
%   Web Address         : gitlab.telecom-paris.fr/elie.awwad/vae-fir/-/blob/main/codes/matlab/lib/moving_average.m?ref_type=heads
%
%   Functions           : moving_std
%   Author              : louis tomczyk
%   Author contact      : louis.tomczyk@telecom-paris.fr
%   Date                : 2024-03-14
%   Title of program    : VAE-FIR
%   Code version        : 1.0.0
%   Type                : source code
%   Web Address         : gitlab.telecom-paris.fr/elie.awwad/vae-fir/-/blob/main/codes/matlab/lib/moving_std.m?ref_type=heads
% ----------------------------------------------
%%

function metrics = get_metrics(caps,thetas,varargin)

if size(thetas.est,2) ~= 1
    thetas.est = reshape(thetas.est,[],1);
end

if size(thetas.gnd,2) ~= 1
    thetas.gnd = reshape(thetas.gnd,[],1);
end

Err.thetas                  = thetas.est-thetas.gnd; % [deg]

metrics.thetas.ErrMedian    = mean(Err.thetas);
metrics.thetas.ErrStd       = std(Err.thetas);
metrics.thetas.ErrRms       = metrics.thetas.ErrStd/metrics.thetas.ErrMedian;
metrics.thetas.Err          = [zeros(caps.NFrames.Training,1);Err.thetas];
params.method               = "mirror";
params.period               = 5;

if params.period < length(metrics.thetas.Err)
    metrics.thetas.Err_mov_avg  = (moving_stat_in(metrics.thetas.Err,params,"average")).';
    metrics.thetas.Err_mov_std  = (moving_stat_in(metrics.thetas.Err,params,"std")).';
end


if ~isempty(varargin)
    phis                                = varargin{1};
Err.phis                                = phis.est.channel-phis.gnd.channel;        % [deg]
if ~strcmpi(caps.rx_mode,'pilots')
% if strcmpi(caps.rx_mode,'pilots') % louis: to remove only if all tests pass
        metrics.phis.ErrMedian          = zeros(caps.log.Nfiles,1);
    else
        metrics.phis.ErrMedian          = zeros(caps.log.Nfiles,3);
    end
    
    metrics.phis.ErrMedian(caps.kdata,:)= median(Err.phis);
    metrics.phis.ErrStd(caps.kdata,:)   = std(Err.phis);
    metrics.phis.ErrRms(caps.kdata,:)   = metrics.phis.ErrStd(caps.kdata,:)./metrics.phis.ErrMedian(caps.kdata,:);

    if ~strcmpi(caps.rx_mode,'pilots')
%     if strcmpi(caps.rx_mode,'pilots')
        metrics.phis.Err                = [zeros(caps.NBatches.Training,1);Err.phis];
    else
        metrics.phis.Err                = Err.phis;
    end
    
    if params.period < length(metrics.phis.Err)
        metrics.phis.Err_mov_avg = (moving_stat_in(metrics.phis.Err,params,"average")).';
        metrics.phis.Err_mov_std = (moving_stat_in(metrics.phis.Err,params,"std")).';
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
% moving_stat_in    (1.1.0)
% ---------------------------------------------

function vector_out = moving_stat_in(vector_in,params,varargin)

% number of elements in the vector to average
N = length(vector_in);

% number of samples used for averaging
M = params.period;

if ~isempty(varargin)
    params.stat = varargin{1};
end

if M == 1
    if strcmpi(params.stat,'std')
        vector_out = std(vector_in);
    elseif strcmpi(params.stat,'average')
        vector_out = vector_in;
    end

    return
end

if strcmp(params.method,'mirror')

    % make a copy of the original vector, flip from LEFT to RIGHT the
    % copy and concatenate along the 2nd axis (columns) both
    vector_in_bis = [vector_in,fliplr(vector_in)];

    % pre-allocate memory space
    vector_out = zeros(1,N);

    % average the original vector over the period and add its value to
    % the new vector
    if strcmpi(params.stat,'std')
        for k=1:length(vector_in)
            vector_out(k) = std(vector_in_bis(k:k+M-1));
        end
    elseif strcmpi(params.stat,'average')
        for k=1:length(vector_in)
            vector_out(k) = 1/M*sum(vector_in_bis(k:k+M-1));
        end
    end

% if other string is used for the method, implentation of bloc
% averaging without adding data.
else
    vector_out = zeros(1,N-M);
    if strcmpi(params.stat,'std')
        for k=1:length(vector_in)-M
            vector_out(k) = std(vector_in(k:k+M-1));
        end
    elseif strcmpi(params.stat,'average')
        for k=1:length(vector_in)-M
            vector_out(k) = 1/M*sum(vector_in(k:k+M-1));
        end
    end

end
% ---------------------------------------------