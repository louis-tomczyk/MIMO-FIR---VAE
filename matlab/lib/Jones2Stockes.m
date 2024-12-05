% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.0
%   Date            : 2024-03-07
%   Arxivs          : 2024-03-04 - creation (1.0.0)
%                   : 2024-03-07 - removing loop in scatter3 + tracking mode
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- Main idea -----
%   Convert the polarisation expression from the Jones formalism
%   to the Stocked one.
%
% ----- INPUTS -----
%   J           (array)     Jones vectors of shape (2,Nstates).
%                           Jones vectors should be unitary.
%   VARARGIN    (cell)      cell array with options: {'track',k,'Nstates',Nstates}
%
% ----- OUTPUTS -----
%   S           (array)         the Stockes vector
%   varargout   (figure handle) used to plot the Poincare sphere. Usefull for superimposing
%                               plots throughout the execution of a script
% ----- EXAMPLE -----
%    S = Jones2Stockes(J,{'track',k,'Nstates',Nstates});
%
% ----- BIBLIOGRAPHY -----
%   Articles/Books
%   Authors             : Jay N. DAMASK
%   Title               : Polarization Optics in Telecommunications
%   Jounal/Editor       : SPRINGER
%   Volume - N°         : 
%   Date                : 2004
%   DOI/ISBN            : 0-387-22493-9
%   Pages               : 54 - 56
% ---------------------------------------------
%%
function [S,varargout] = Jones2Stockes(J,varargin)

if size(J,1) ~= 2
    J = reshape(J,2,[]);
end

pauli_1 = [1,0; 0,-1];
pauli_2 = [0,1; 1,0];
pauli_3 = [0,-1; 1,0]*1i;

Nstates = size(J,2);
S       = zeros(4,Nstates);

for k = 1:Nstates
    S1(k)      = J(:,k)'*pauli_1*J(:,k);
    S2(k)      = J(:,k)'*pauli_2*J(:,k);
    S3(k)      = J(:,k)'*pauli_3*J(:,k);

    S0(k)      = sqrt(S1(k).^2+S2(k).^2+S3(k).^2);
    S(:,k)     = [S0(k),S1(k),S2(k),S3(k)]/S0(k).';
end

% condition = sum(round(S(1,:),7)+eps == round(ones(1,Nstates),7)+eps)/Nstates;
% assert(condition, ...
% 'Stockes vector should be normalised, i.e. S_0 = 1')

S = S(2:4,:);

if nargin == 1
    varargout{1} = plot_Poincare_state(S,Nstates);
else
    if ishandle(1)
        plot_Poincare_state(S,Nstates,varargin{1});
    else
        varargout{1} = plot_Poincare_state(S,Nstates,varargin{1});
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   plot_Poincare_state     (S,Nstates)
% ---------------------------------------------

function varargout = plot_Poincare_state(S,Nstates,varargin)

if ~ishandle(1)
    varargout{1}= figure(001);
    hold on
    [x, y, z]   = sphere();
    h           = surf(x, y, z,FaceColor=[1,1,1]*0.73,EdgeColor='none'); 

    S1_Xaxis    = [-1,1]*1.2;
    S1_Yaxis    = [0,0];
    S1_Zaxis    = [0,0];

    S2_Xaxis    = [0,0];
    S2_Yaxis    = [-1,1]*1.2;
    S2_Zaxis    = [0,0];

    S3_Xaxis    = [0,0];
    S3_Yaxis    = [0,0];
    S3_Zaxis    = [-1,1]*1.2;

    R       = ones(50,1);
    Theta   = linspace(-pi,pi,50)';

    ZS1S2   = 0*R;
    XS1S2   = R.*cos(Theta);
    YS1S2   = R.*sin(Theta);

    ZS1S3   = R.*cos(Theta);
    XS1S3   = R.*sin(Theta);
    YS1S3   = 0*R;

    ZS2S3   = R.*cos(Theta);
    XS2S3   = 0*R;
    YS2S3   = R.*sin(Theta);

    plot3(S1_Xaxis,S1_Yaxis,S1_Zaxis,LineWidth=2,Color='k')
    plot3(S2_Xaxis,S2_Yaxis,S2_Zaxis,LineWidth=2,Color='k')
    plot3(S3_Xaxis,S3_Yaxis,S3_Zaxis,LineWidth=2,Color='k')
    plot3(XS1S2,YS1S2,ZS1S2,'-.',LineWidth=1,Color='k')
    plot3(XS1S3,YS1S3,ZS1S3,'-.',LineWidth=1,Color='k')
    plot3(XS2S3,YS2S3,ZS2S3,'-.',LineWidth=1,Color='k')
   
    text(1.3,0,0,"S_1","fontsize",15,"fontweight","bold","fontname","Times",Interpreter="tex")
    text(0,1.3,0,"S_2","fontsize",15,"fontweight","bold","fontname","Times",Interpreter="tex")
    text(0,0,1.3,"S_3","fontsize",15,"fontweight","bold","fontname","Times",Interpreter="tex")
end

hold on
S       = 1.02*S;
colors  = hot(Nstates);
if nargin > 2

    myfill      = varargin{1}{1+find([varargin{1}{:}] == 'fill')};
    mymarker    = varargin{1}{1+find([varargin{1}{:}] == 'marker')};
    mysize      = varargin{1}{1+find([varargin{1}{:}] == 'size')};

    if isempty(myfill) || ~strcmpi(myfill(1:4),'fill')
        scatter3(S(1,:),S(2,:),S(3,:),mysize,colors,mymarker);
    else
        scatter3(S(1,:),S(2,:),S(3,:),mysize,colors,'filled',mymarker);
    end
end


axis equal
%     view(155,21)
view(63,15)
title("Poincare Sphere")
grid off
set(gca,"fontsize",15,"fontweight","bold","fontname","Times")
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
set(gca,'ZTickLabel',[]);

axis off

