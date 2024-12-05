% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.0
%   Date            : 2024-03-08
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- Main idea -----
%   Get the Stockes vector from the FIR
%
% ----- INPUTS -----
%   FIR     (struct)        the FIR weigths FIR = [HH, HV; VH, VV]
%               - (a,b)     in {H,V}^2, the weights of (ab) FIR coefficients 
%                           FIR.ab  (array)     size = Nframes x Ntaps, see python/main.py
%   VARARGIN(cell)          cell array with options: {'track',k,'Nstates',Nstates}
% 
% ----- OUTPUTS -----
%   S       (array)         the Stockes vector
%   f       (figure handle) used to plot the Poincare sphere. Usefull for superimposing
%                           plots throughout the execution of a script
%
% ----- EXAMPLE -----
% params = {"marker"  ,'o'        ,...
%           "size"    , 50        ,...
%           "fill"    , 'filled'};
% [Sest,f]   = FIR2Stockes(FIRest,params);
%
% ----- BIBLIOGRAPHY -----
%   Articles/Books
%   Authors             : J.C.Geyer et. al.
%   Title               : Channel parameter estimation for polarisation diverse coherent receivers
%   Jounal/Editor       : Photonics Technology Letters
%   Volume - N°         : 20-10
%   Date                : 2008-05-15
%   DOI/ISBN            : 10.1109/LPT.2008.21104
%   Pages               : 776-778
% ---------------------------------------------
%%
function [S,f] = FIR2Stockes(FIR,varargin)

[JH,~] = FIR2Jones(FIR);

if nargin == 1
    [S(:,:,1),f]    = Jones2Stockes(JH);
%         S(:,:,2)        = Jones2Stockes(JV);
else
    if ishandle(1)
        S(:,:,1)    = Jones2Stockes(JH,varargin{1});
%             S(:,:,2)    = Jones2Stockes(JV,varargin{1});
    else
        [S(:,:,1),f]= Jones2Stockes(JH,varargin{1});
%             S(:,:,2)    = Jones2Stockes(JV,varargin{1});
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   FIR2Jones       FIR         J
% ---------------------------------------------

function [JH,JV] = FIR2Jones(FIR)

Nstates = size(FIR.HH,1);

if size(FIR.HH,2) == 1 % one tap perF FIR
    Hinv   = zeros(2,2,Nstates);
    for k = 1:Nstates
        Hinv(:,:,k) = [FIR.HH(k), FIR.HV(k);...
                       FIR.VH(k), FIR.VV(k)];
    end
    H   = pageinv(Hinv);
    JH  = squeeze([H(1,1,:); H(2,1,:)]);
    JV  = squeeze([H(1,2,:); H(2,2,:)]);
else
    JH  = [sum(FIR.HH,2), sum(FIR.VH,2)]';
    JV  = [sum(FIR.HV,2), sum(FIR.VV,2)]';
end

%% ---------------------------------------------