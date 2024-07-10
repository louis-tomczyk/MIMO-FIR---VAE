% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Arxivs          :
%   Date            : 2024-07-10
%   Version         : 1.0.2
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-06 (1.0.0) creation
%   2024-07-09 (1.0.1) caps: closing figures + number of Frames
%   2024-07-10 (1.0.2) caps: sorting the struct
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

function caps = extract_infos(caps,Dat,kdata)


data                = Dat{kdata};
data                = sort_struct_alphabet(data);
filename            = char(caps.Fn{kdata});
caps.filename       = ['matlab',filename(1:end-4)];



caps.NFrames        = double(data.NFrames);
caps.FrameChannel   = double(data.FrameChannel);
caps.NFramesTraining= double(data.NFramesTraining);
caps.Frames         = linspace(1,caps.NFramesTraining,caps.NFramesTraining);
caps.NBatchFrame    = double(data.NBatchFrame);
caps.NBatchesChannel= double(data.NBatchesChannel);
caps.FIRlength      = double(data.NspTaps);
caps.FIRtaps        = linspace(1,caps.FIRlength,caps.FIRlength)-caps.FIRlength/2;
caps.kdata          = kdata;
caps.NFramesChannel = caps.NFrames-caps.FrameChannel;

if length(caps.Fn) > 1
    caps.flags.close = 1;
else
    caps.flags.close = 0;
end

caps = sort_struct_alphabet(caps);

end