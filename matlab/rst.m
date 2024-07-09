%%
% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Arxivs          :
%   Date            : 2024-07-09
%   Version         : 1.0.0
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
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

clear all
close all
clc

default_plots()
addpath ./lib
currentDate = datetime('now');
Date        = datestr(currentDate, 'yy-mm-dd');
myRootPath  = pwd();



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----             
%   default_plots
% ---------------------------------------------

function default_plots()
    set(groot,'defaultAxesTickLabelInterpreter','latex'); 
    set(groot,'defaulttextinterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');
    set(groot,'defaultFigureUnits','normalized')
    set(groot,'defaultFigurePosition',[0 0 1 1])
    set(groot,'defaultAxesFontSize', 18);
    set(groot,'defaultfigurecolor',[1 1 1])
    set(groot,'defaultAxesFontWeight', 'bold')
end