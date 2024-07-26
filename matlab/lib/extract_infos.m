% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.0
%   Date            : 2024-07-26
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-06  (1.0.0) creation
%   2024-07-09  (1.0.1) caps: closing figures + number of Frames
%   2024-07-10  (1.0.2) caps: sorting the struct
%   2024-07-11  (1.0.3) phase noise management
%   2024-07-12  (1.0.4) phase noise management --- for rx['mode'] = 'pilots'
%   2024-07-16  (1.0.5) multiple files processing
%   2024-07-23  (1.0.6) ThStd -> Th_std
%   2024-07-25  (1.0.7) custom what_carac
%   2024-07-26  (1.1.0) removed what_carac, along with main_0_python2matlab (2.1.1)
%                       [REMOVED] nested functions
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

function caps = extract_infos(caps,data)

% logistics
data                    = sort_struct_alphabet(data);
filename                = char(caps.log.Fn{caps.kdata});
caps.log.filename       = ['matlab',filename(1:end-4)];

if length(caps.log.Fn) > 1
    caps.plot.close     = 1;
else
    caps.plot.close     = 0;
end


caps.rx_mimo            = data.rx_mimo;
caps.rx_mode            = data.rx_mode;

% frames
caps.Frames.Channel     = double(data.FrameChannel);
caps.NFrames.Training   = double(data.NFramesTraining);
caps.NFrames.all        = double(data.NFrames);
caps.NFrames.Channel    = caps.NFrames.all-caps.NFrames.Training;
caps.NFrames.Channel    = caps.NFrames.all-caps.Frames.Channel;
caps.Frames.array       = linspace(1,caps.NFrames.Channel,caps.NFrames.Channel);

% batches
caps.NBatches.Frame     = double(data.NBatchesFrame);
caps.NBatches.Channel   = double(data.NBatchesChannel);

if isfield(data, 'Phis_gnd') && ~sum(sum(isnan(data.Phis_gnd)))
    if ~strcmpi(caps.rx_mode,'blind') || strcmpi(caps.rx_mimo,'vae')
        caps.phis_est   = 1;
    else
        caps.phis_est   = 0;
    end
else
    caps.phis_est       = 0;
end


if strcmpi(caps.rx_mode,'blind')
    caps.NBatches.Training  = caps.NFrames.Training*caps.NBatches.Frame;
    caps.NBatches.Channel   = caps.NBatches.Frame*caps.NFrames.Channel;
else
    caps.NBatches.FrameCut  = caps.NBatches.Frame-2;
    caps.NBatches.Channel   = caps.NBatches.FrameCut*caps.NFrames.Channel;
    caps.NBatches.Training  = caps.NBatches.FrameCut*caps.NFrames.Training;
end



if caps.phis_est
    caps.Batches.array      = linspace(1,caps.NBatches.Channel,caps.NBatches.Channel);
else
    caps.Batches.array      = NaN;
end


if ~isfield(caps.plot.phis,'pol')
    caps.plot.phis.pol      = 1;
end

% fir
caps.FIR.length             = double(data.NspTaps);
caps.FIR.tap                = ceil(caps.FIR.length/2);
caps.FIR.taps               = linspace(1,caps.FIR.length,caps.FIR.length)-caps.FIR.tap;


cd(caps.log.PathSave{1})
caps = sort_struct_alphabet(caps);

end

