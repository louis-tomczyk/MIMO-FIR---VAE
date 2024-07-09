% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Arxivs          :
%   Date            : 2024-07-06
%   Version         : 1.0.0
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
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
filename            = char(caps.Fn{kdata});
caps.filename       = ['matlab',filename(1:end-4)];
data.FIRlength      = double(data.NtapsTX);


caps.Nframes        = double(data.Nframes);
caps.FrameChannel   = double(data.FrameChannel);
caps.frames         = linspace(1,caps.Nframes-caps.FrameChannel,caps.Nframes-caps.FrameChannel);
caps.FIRlength      = double(data.NtapsTX);
caps.FIRtaps        = linspace(1,caps.FIRlength,caps.FIRlength)-caps.FIRlength/2;
caps.kdata          = kdata;

if length(caps.Fn) > 1
    caps.flags.close = 1;
else
    caps.flags.close = 0;
end

end