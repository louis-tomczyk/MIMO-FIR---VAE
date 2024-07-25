% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.1.0
%   Date            : 2024-07-25
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
%   2024-07-06  (2.0.0) encapsulation into modules
%                       [REMOVED] check_if_fibre_prop
%   2024-07-09  (2.0.1) phase estimation
%   2024-07-10  (2.0.2) flexibility and naming standardisation
%   2024-07-11  (2.0.3) cleaning caps structure 
%   2024-07-12  (2.0.4) phase noise management --- for rx['mode'] = 'pilots'
%                       import_data: caps structuring
%   2024-07-16  (2.0.5) multiple files processing
%   2024-07-19  (2.0.6) import_data: managing files not containing data
%   2024-07-23  (2.0.7) progression bar
%   2024-07-25  (2.1.0) import_data (1.1.0): finner selection of files
%                       wrong values saved in <Err *> corrected
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
Th_in   = [0,5,10,15,20,25,45];
fpol    = [6,18,57,180];

% caps.log.Date = '24-07-23';

for nth = 1:length(Th_in)
    for nfpol = 1:length(fpol)
        cd(strcat('../python/data-',caps.log.Date,"/mat"))
        caps.log.myInitPath     = pwd();
        
        selected_caracs         = [sprintf("Th_in %.1f",Th_in(nth));sprintf("fpol %.1f",fpol(nfpol))];
        [allData,caps]          = import_data({'.mat'},caps,selected_caracs); % {,manual selection}
        cd(caps.log.myInitPath)
        
        caps.what_carac         = "Th_in";
        
        caps.plot.fir           = 0;
        caps.plot.poincare      = 0;
        caps.plot.SOP.xlabel    = 'comparison per frame';   % {'error per frame','error per theta''comparison per frame'}
        caps.plot.phis.xlabel   = 'comparison per batch';
        caps.method.thetas      = 'mat';                    % {fft, mat, svd}
        caps.method.phis        = 'eig';
        caps.save.mean_errs     = 1;
        
        
        if length(allData) > 5 f = waitbar(0, 'Starting'); end
        
        for kdata = 1:length(allData)
        
            data                        = allData{kdata};
            caps.kdata                  = kdata;
            caps                        = extract_infos(caps,data);
            [caps,thetas,phis, H_est]   = channel_estimation(data,caps);
            [thetas, phis]              = extract_ground_truth(data,caps,thetas,phis);
            
            if caps.phis_est
                metrics                 = get_metrics(caps,thetas,phis);
                plot_results(caps,H_est, thetas,metrics,phis);
            else
                metrics                 = get_metrics(caps,thetas);
                plot_results(caps,H_est, thetas,metrics);
            end
        
            cd ../err/thetas
        
            if kdata == 1
                Mthetas         =  zeros(caps.log.Nfiles+1,caps.carac.Ncarac+3);
                if caps.phis_est
                    Mthetas     =  zeros(caps.log.Nfiles+1,caps.carac.Ncarac+3);
                end
            end
        
            Mthetas(kdata,:)    = [caps.carac.values(kdata,:),...
                                   metrics.thetas.ErrMean,...
                                   metrics.thetas.ErrStd,...
                                   metrics.thetas.ErrRms];
        
            if caps.phis_est
                cd ../phis
                Mphis(kdata,:)  = [caps.carac.values(kdata,:),...
                                   metrics.phis.ErrMean(end),...
                                   metrics.phis.ErrStd(end),...
                                   metrics.phis.ErrRms(end)];
        
            end
        
            if length(allData) > 5
                waitbar(kdata/length(allData), f,...
                    sprintf('Progress: %d %%', floor(kdata/length(allData)*100)));
            end
        end
        if length(allData) > 5 close(f); end
        
        
        if caps.save.mean_errs 
            cd ../thetas
            Mthetas(end,:)  = [caps.carac.values(kdata,:),...
                               median(Mthetas(1:end-1,end-2)),...
                               median(Mthetas(1:end-1,end-1)),...
                               median(Mthetas(1:end-1,end))];
            writematrix(Mthetas,strcat('<Err Theta>-',caps.log.filename,'.csv'))
        
            if caps.phis_est
                cd ../phis
                Mphis(end,:)= [caps.carac.values(kdata,:),...
                               median(Mphis(1:end-1,end-2)),...
                               median(Mphis(1:end-1,end-1)),...
                               median(Mphis(1:end-1,end))];
                writematrix(Mphis,strcat('<Err Phi>-',caps.log.filename,'.csv'))
            end
        end
        
        
        
        cd(caps.log.myRootPath)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----             
%   import_data         (1.1.0)
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
    
    elseif nargin == 2
        % User interface
        [filenames, pathname] = uigetfile(...
            strcat('*', acceptedFormats),...
            'Select files',...
            'MultiSelect', 'on');
    else
        pathname    = pwd;                                      % Get current directory
        filenames   = dir(fullfile(pathname, '*.*'));           % Get all files
        filenames   = {filenames.name};                         % Extract file names
        filenames   = filenames(~startsWith(filenames, '.'));   % Exclude hidden files
        for k = 1:length(varargin{1})
            filenames   = filenames(contains(filenames, varargin{1}(k)));
            % Exclude files not containing the specific keywords given in varargin
        end
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
        else
            allFilenames{i} = filenames{i};
            allPathnames{i} = pathname;
            allData{i}      = NaN;
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

