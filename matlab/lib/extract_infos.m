% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Date            : 2024-07-15
%   Version         : 1.0.5
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
%   2024-07-15  (1.0.5) multiple files processing
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
    if ~(strcmpi(caps.rx_mimo ,'cma') && strcmpi(caps.rx_mode,'blind'))
        caps.phis_est           = 1;
        caps.Batches.array      = linspace(1,caps.NBatches.Channel,caps.NBatches.Channel);
    else
        caps.phis_est           = 0;
        caps.Batches.array      = NaN;
    end
else
    caps.phis_est           = 0;
    caps.Batches.array      = NaN;
end


if ~strcmpi(caps.rx_mode,'pilots')
    caps.NBatches.Training  = caps.NFrames.Training*caps.NBatches.Frame;
    caps.NBatches.Channel   = caps.NBatches.Frame*caps.NFrames.Channel;

else
    caps.NBatches.FrameCut  = caps.NBatches.Frame-2;
    caps.NBatches.Channel   = caps.NBatches.FrameCut*caps.NFrames.Channel;
    caps.NBatches.Training  = caps.NBatches.FrameCut*caps.NFrames.Training;

end


if ~isfield(caps.plot.phis,'pol')
    caps.plot.phis.pol = 1;
end

% fir
caps.FIR.length         = double(data.NspTaps);
caps.FIR.taps           = linspace(1,caps.FIR.length,caps.FIR.length)-ceil(caps.FIR.length/2);


% caracs
what_carac         = string({''});

if strcmpi(data.ThLaw,'lin')
    what_carac    = [what_carac,string({'ThEnd','Sth'})];
elseif strcmpi(data.ThLaw,'gauss')
    disp("Warning: all possible models implemented in Python's simulator" + ...
        "are not taken into account. If not LINEAR model, Rwalk-Gauss by" + ...
        "default is considered. Future versions may need to take it into" + ...
        "account")
    what_carac    = [what_carac,string({'Thstd'})];  % {dnu, Sth, Sph, ThEnd, PhEnd,Thstd, Phstd}
end


if caps.phis_est && strcmpi(data.PhLaw,'lin') && ~sum(sum(isnan(data.Phis_gnd)))
    what_carac    = [what_carac,string({'PhEnd','Sph'})];
elseif strcmp(data.PhLaw,'Linewidth')
    what_carac    = [what_carac,string({'dnu'})];
end

caps.carac.what     = what_carac(2:end);

caps.carac.values   = zeros(length(caps.log.Fn),length(caps.carac.what));
for k = 1:length(caps.carac.what)
    caps.carac.values(:,k)   = get_value_from_filename_in(caps.log.PathSave,caps.carac.what{k},caps.log.Fn);
end

caps.carac.Ncarac   = length(caps.carac.what);

caps = sort_struct_alphabet(caps);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   get_value_from_filename_in
%   get_number_from_string_in
% ---------------------------------------------


function out = get_value_from_filename_in(folderPath,quantity,varargin)

    cd(folderPath{1})
  
    if nargin == 2
        nfiles          = length(dir(pwd))-2;
        folder_struct   = dir(pwd);
        out             = zeros(nfiles,1);

        for k=1:nfiles
            filename    = folder_struct(k+2).name;
            out(k)      = get_number_from_string_in(filename,quantity);
        end

    else
        nfiles          = length(varargin{1});
        out             = zeros(nfiles,1);
        for k=1:nfiles
            out(k)      = get_number_from_string_in(varargin{1}{k},quantity);
        end
    end

    out = sort(out);
    
end
%-----------------------------------------------------



function out = get_number_from_string_in(stringIn,what,varargin)

    stringIn    = char(stringIn);
    iwhat       = strfind(stringIn,what);

    if nargin == 2
        iendwhat    = iwhat+length(what);
        idashes     = strfind(stringIn,'-');
        [~,itmp]    = max(idashes-iendwhat>0);
        idashNext   = idashes(itmp);
        strTmp      = stringIn(iendwhat+1:idashNext-1);
    else
        if nargin > 2
            if iwhat-varargin{1}<1
                istart = 1;
            else
                istart = iwhat-varargin{1};
            end
            if nargin == 4
                if iwhat+varargin{2}>length(stringIn)
                    iend = length(stringIn);
                else
                    iend = iwhat+varargin{2};
                end
            end
            strTmp  = stringIn(istart:iend);
        end
    end

    indexes = regexp(strTmp,'[0123456789.]');
    out     = str2double(strTmp(indexes));
end
%-----------------------------------------------------