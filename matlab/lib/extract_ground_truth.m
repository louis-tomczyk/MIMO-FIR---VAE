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

function thetas_gnd = extract_ground_truth(Dat,caps)

    data        = Dat{caps.kdata};
    h_gnd       = extract_Hgnd(data);
    
    thetas_gnd  = zeros(3,data.Nframes-data.FrameChannel);
    h11s        = zeros(data.Nframes-data.FrameChannel,1);
    h12s        = zeros(data.Nframes-data.FrameChannel,1);
    h21s        = zeros(data.Nframes-data.FrameChannel,1);
    h22s        = zeros(data.Nframes-data.FrameChannel,1);
    
    for k = 1:data.Nframes-data.FrameChannel
        h11s(k)         = h_gnd.(sprintf('frame%i',k+data.FrameChannel)).h11;
        h12s(k)         = h_gnd.(sprintf('frame%i',k+data.FrameChannel)).h12;
        h21s(k)         = h_gnd.(sprintf('frame%i',k+data.FrameChannel)).h21;
        h22s(k)         = h_gnd.(sprintf('frame%i',k+data.FrameChannel)).h22;
    
        ratio_hij_H     = abs(h12s(k)/h11s(k));
        ratio_hij_V     = abs(h21s(k)/h22s(k));
    
        thetas_gnd(1,k) = atan(ratio_hij_H)*180/pi;   % [deg]
        thetas_gnd(2,k) = atan(ratio_hij_V)*180/pi;   % [deg]
    end
    
    thetas_gnd(3,:)     = mean(thetas_gnd(1:2,:));  % [deg]
    FIRgnd.HH           = h11s;
    FIRgnd.HV           = h12s;
    FIRgnd.VH           = h21s;
    FIRgnd.VV           = h22s;
    
    if caps.flags.poincare
        params = {"marker"  ,'square'   ,...
              "size"    , 100        ,...
              "fill"    , ''};
        Sgnd   = FIR2Stockes(FIRgnd,params);
    end
    
    
    
    
    
    
    
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   check_if_fibre_prop
%   extract_ground_truth
% ---------------------------------------------


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
%-----------------------------------------------------





function bool = check_if_fibre_prop(h)

    input = h.frame1.h11{1};
    
    if all(input == input(1),'all')
        bool = false;
    else
        bool = true;
    end

%-----------------------------------------------------