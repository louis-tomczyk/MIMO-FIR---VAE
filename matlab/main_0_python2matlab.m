% ---------------------------------------------
% ----- INFORMATIONS -----
%   Function name   : processing_0_python2matlab
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Date            : 2024-07-12
%   Version         : 2.0.4
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2023-10-09  (1.0.0)
%   2024-03-04  (1.1.0) [NEW] plot poincare sphere
%   2024-03-29  (1.1.1) data.FrameChannel -> data.FrameChannel
%   2024-04-18  (1.1.3) import_data
%   2024-04-19  (1.1.4) <Err Theta>
%   ------------------
%   2024-07-06  
% (2.0.0) encapsulation into modules
%                       [REMOVED] check_if_fibre_prop
%   2024-07-09  (2.0.1) phase estimation
%   2024-07-10  (2.0.2) flexibility and naming standardisation
%   2024-07-11  (2.0.3) cleaning caps structure 
%   2024-07-12  (2.0.4) phase noise management --- for rx['mode'] = 'pilots'
%                       import_data: caps structuring
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
rst

cd(strcat('../python/data-',caps.log.Date,"/mat"))
caps.log.myInitPath     = pwd();
[allData,caps]          = import_data({'.mat'},caps,'manual selection'); % {,manual selection}
cd(caps.log.myInitPath)
caps.plot.fir           = 1;
caps.plot.poincare      = 0;
caps.plot.phis.do       = 1;
caps.plot.SOP.xlabel    = 'comparison per frame';   % {'error per frame','error per theta''comparison per frame'}
caps.plot.phis.xlabel   = 'comparison per batch';
caps.method.thetas      = 'fft';                    % {fft, mat, svd}
caps.method.phis        = 'eig';


for tap = 7:7

    caps.FIR.tap = tap;

    for kdata = 1:length(allData)
    
        data                        = allData{kdata};
        caps.kdata                  = kdata;
        caps                        = extract_infos(caps,data);
        [caps,thetas,phis, H_est]   = channel_estimation(data,caps);
        [thetas, phis]              = extract_ground_truth(data,caps,thetas,phis);
        
        if caps.plot.phis.do
            metrics             = get_metrics(caps,thetas,phis);
            plot_results(caps,H_est, thetas,metrics,phis);
        else
            metrics             = get_metrics(caps,thetas);
            plot_results(caps,H_est, thetas,metrics);
        end

    end
end

cd ../err
tmp                 = zeros(size(caps.carac.values));
Mthetas             = [caps.carac.values,...
                       metrics.thetas.ErrMean,...
                       metrics.thetas.ErrStd,...
                       metrics.thetas.ErrRms];
Mthetas(end+1,:)    = [tmp,...
                       median(metrics.thetas.ErrMean),...
                       median(metrics.thetas.ErrStd),...
                       median(metrics.thetas.ErrRms)];

writematrix(Mthetas,strcat('<Err Theta>-',caps.log.filename,'.csv'))

% (end), for rx_mode = pilots
if caps.plot.phis.do
    Mphis           = [caps.carac.values,...
                       metrics.phis.ErrMean(end),...
                       metrics.phis.ErrStd(end),...
                       metrics.phis.ErrRms(end)];

    Mphis(end+1,:)  = [tmp,...
                       median(metrics.phis.ErrMean(end)),...
                       median(metrics.phis.ErrStd(end)),...
                       median(metrics.phis.ErrRms(end))];

    writematrix(Mphis,strcat('<Err Phi>-',caps.log.filename,'.csv'))
end

cd(caps.log.myRootPath)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----             
%   import_data
%   get_value_from_filename_in
%   get_number_from_string_in
% ---------------------------------------------

function [allData, caps] = import_data(acceptedFormats,caps,varargin)
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

    caps.log.Fn         = allFilenames;
    caps.log.PathSave   = allPathnames;
    caps.log.Nfiles     = length(allData);
end
%-----------------------------------------------------
