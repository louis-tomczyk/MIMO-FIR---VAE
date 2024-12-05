% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.0.0
%   Date            : 2024-11-13
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- MAIN IDEA -----
%   Unwrap when Matlab unwrapping fails because angle shifts
%   are below pi/2
%
% ----- INPUTS -----
% ----- OUTPUTS -----
% ----- EXAMPLE -----
%    S = Jones2Stockes(J,{'track',k,'Nstates',Nstates});
%
% ----- BIBLIOGRAPHY -----
%   Articles/Books
%   Authors             :
%   Title               :
%   Jounal/Editor       :
%   Volume - N°         : 
%   Date                :
%   DOI/ISBN            :
%   Pages               :
% ---------------------------------------------
%%

function y = handmade_unwrap(x,varargin)

    if ~isempty(varargin)
      ref       = 90/varargin{1};
    else
      ref       = 90
    end

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