% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.0.0
%   Date            : 2024-04-19
%   Licence         : cc-by-nc-sa
%                     Attribution - Non-Commercial - Share Alike 4.0 International
%
% ----- MAIN IDEA -----
%   See VAE ability to tract the State of Polarisation
%   of a beam propagating into an optical fibre.
%
% ----- INPUTS -----
% ----- BIBLIOGRAPHY -----
%   Functions           :
%   Author              : Diane PRATO
%   Author contact      : diane.prato@telecom-paris.fr
%   Date                : 2023-06
%   Title of program    : plot_H
%   Code version        : 1.0
%   Type                : 
%   Web Address         : 
% ----------------------------------------------
%%

%% MAINTENANCE
% rst
% default_plots()
% addpath '/home/louis/L_Libraries/Matlab/louis library/general'
% addpath '/home/louis/L_Libraries/Matlab/louis library/mathematics/'

myInitPath      = pwd();
% cd ../data/data-ECOC/Rs64/PolLaw_Rwalk/Fpol10000/mat/vae/
% cd ../data/data-ECOC/Rs64/SNRs/mat/vae/SNR28
% cd ../data/data-ECOC/Rs64/PolLaw_Rwalk/Fpol100/mat/cma/
% cd '../python/data-2024-04-18/128QAM/17 040/15-75/errors/'
cd ../data/data-ECOC/Rs64/PolLaw_Lin/

[Dat,Fn,PathSave]   = import_data({'.csv'},'manual selection');
% [Dat,Fn,PathSave]   = import_data({'.csv'});
Nfiles              = length(Dat);
myPath              = PathSave{1};
cd(myInitPath)

errors_mean         = zeros(Nfiles,1);
errors_std          = zeros(Nfiles,1);
ThetaStart         = zeros(Nfiles,1);

for k = 1:Nfiles
    Dat{k};
    ThetaStart(k)   = 90-Dat{k}.Var1(1);
    errors_mean(k)  = abs(Dat{k}.Var2(end));
    errors_std(k)   = abs(Dat{k}.Var3(end));
end
data = sortrows([ThetaStart,errors_mean,errors_std],1);


colnames        = {'ThetaStart',  'errorMean','errorStd'};
myTable         = array2table(data,"VariableNames",colnames);

Carac           = get_value_from_filename_in(PathSave{1},'Slope');
tmp             = PathSave{1};
fname_saved     = tmp(end-14:end);
fname_saved     = replace(fname_saved,'/','_');
% fname_saved     = strcat(fname_saved);
writetable(myTable,fname_saved)


shaded_area_y_top   = 0.05*data(:,1);
shaded_area_y_bot   = -0.05*data(:,1);
y = [shaded_area_y_top,shaded_area_y_bot];

createfigure(data(:,1),data(:,2),data(:,3),y,'square')

legend('$\pm 5$\%','','64 [GBd] - $\Delta\theta$ = 0.5 [deg/frame]')

% cd(PathSave{1})


cd(myInitPath)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   default_plots               
%   import_data                 acceptedFormats                 [allData, allFilenames, allPathnames]
%   get_value_from_filename_in  folderPath,quantity,varargin    out
%   get_number_from_string_in   stringIn,what,varargin          out
%   check_if_fibre_prop         h                               bool
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
%-----------------------------------------------------

function [allData, allFilenames, allPathnames] = import_data(acceptedFormats,varargin)
    if nargin < 1
        acceptedFormats = {};
    end

    if ~iscell(acceptedFormats)
        error("'acceptedFormats' must be a cell array.");
    end

    acceptedFormats = string(acceptedFormats);

    % Automatically select all files in current directory if no formats specified
    if nargin == 0 || isempty(acceptedFormats)
        pathname    = pwd;                                      % Get current directory
        filenames   = dir(fullfile(pathname, '*.*'));           % Get all files
        filenames   = {filenames.name};                         % Extract file names
        filenames   = filenames(~startsWith(filenames, '.'));   % Exclude hidden files
    
    elseif nargin == 1
        pathname    = pwd;                                      % Get current directory
        filenames   = dir(fullfile(pathname, strcat('*', acceptedFormats)));           % Get all files
        filenames   = {filenames.name};                         % Extract file names
        filenames   = filenames(~startsWith(filenames, '.'));   % Exclude hidden files
        filenames   = filenames(~startsWith(filenames, 'SUM_UP'));   % Exclude hidden files
    else
        % User interface
        [filenames, pathname] = uigetfile(...
            strcat('*', acceptedFormats),...
            'Select files',...
            'MultiSelect', 'on');
    end



    % Convert to cell array if needed
    if ~iscell(filenames)
        filenames = {filenames};
    end

    % Sorting the files
    Nfiles  = length(filenames);
    tmp     = strings(Nfiles,1);

    for k = 1:Nfiles
        tmp(k) = filenames{k};
    end
    tmp     = sort_strings(tmp);

    for k = 1:Nfiles
        filenames{k} = tmp(k);
    end

    allData         = cell(1,Nfiles);
    allFilenames    = cell(1,Nfiles);
    allPathnames    = cell(1,Nfiles);
    
    for i = 1:Nfiles
        % [1] file path construction
        selectedFile = fullfile(pathname, filenames{i});

        % [2] file name extraction from file path
        [~, ~, fileExtension] = fileparts(selectedFile);

        % [3] check if no extension specified, load all files
        if isempty(acceptedFormats)
            data            = readtable(selectedFile);
            allData{i}      = data;
            allFilenames{i} = filenames{i};
            allPathnames{i} = pathname;
            continue;
        end

        % [4] file extension check
        if ~any(char(acceptedFormats) == fileExtension)
            disp(['File format not supported for file: ' filenames{i}]);
            continue
        end

        % [5] data loading
        switch lower(fileExtension)
            case '.txt'
                data = load(selectedFile);
                % If headers:
                % data = readtable(selectedFile);
            case '.csv'
                data = readtable(selectedFile);
            case '.mat'
                data = load(selectedFile);
            otherwise
                disp(['File format not supported for file: ' filenames{i}]);
                continue
        end

        allData{i}      = data;
        allFilenames{i} = filenames{i};
        allPathnames{i} = pathname;
    end
end
%-----------------------------------------------------

function out = get_value_from_filename_in(folderPath,quantity,varargin)

    cd(folderPath)
  
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





function createfigure(X1, Y1, D1, ymatrix1,myMarker)
%CREATEFIGURE(X1, Y1, D1, ymatrix1)
%  X1:  errorbar x vector data
%  Y1:  errorbar y vector data
%  D1:  errorbar delta vector data
%  YMATRIX1:  area matrix data

%  Auto-generated by MATLAB on 19-Apr-2024 17:14:28


figure1 = figure;
hold on
area(X1,ymatrix1,FaceColor=[1,1,1]*0.83,EdgeColor = [1,1,1]*0.83,FaceAlpha=0.3)
area(X1,-ymatrix1,FaceColor=[1,1,1]*0.83,EdgeColor = [1,1,1]*0.83,FaceAlpha=0.3,HandleVisibility='off')


% Create errorbar
errorbar(X1,Y1,D1,...
    'LineStyle','none',...
    'LineWidth',2,...
    'Marker',myMarker,...
    'Color',[0 0 0]);



% Create ylabel
ylabel('estimation  error = $\langle |\hat{\theta}-\theta| \rangle$ [deg]','FontWeight','bold');



% Create xlabel
xlabel('$\theta_{start}$ [deg]','FontWeight','bold');



end