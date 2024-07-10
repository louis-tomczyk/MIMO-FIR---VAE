% ---------------------------------------------
% ----- INFORMATIONS -----
%   Function name   : processing_0_python2matlab
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Date            : 2024-07-10
%   Version         : 2.0.2
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2023-10-09 (1.0.0)
%   2024-03-04 (1.1.0) [NEW] plot poincare sphere
%   2024-03-29 (1.1.1) data.FrameChannel -> data.FrameChannel
%   2024-04-18 (1.1.3) import_data
%   2024-04-19 (1.1.4) <Err Theta>
%   ------------------
%   2024-07-06 (2.0.0)  encapsulation into modules
%                       [REMOVED] check_if_fibre_prop
%   2024-07-09 (2.0.1)  phase estimation
%   2024-07-10 (2.0.2)  flexibility and naming standardisation
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

cd(strcat('../python/data-',Date,"/mat"))
caps.myInitPath     = pwd();

[Dat,caps]          = import_data({'.mat'},caps,'manual selection'); % {,manual selection}

Nfiles              = length(Dat);
ErrMean             = zeros(Nfiles,1);
ErrStd              = zeros(Nfiles,1);
ErrRms              = zeros(Nfiles,1);
what_carac          = 'dnu';  % {dnu, Slope, End,std}
Carac               = get_value_from_filename_in(caps.PathSave,what_carac,caps.Fn);
cd(caps.myInitPath)


caps.flags.fir          = 1;
caps.flags.poincare     = 0;
caps.flags.SOP          = 'comparison per frame';   %{'error per frame','error per theta''comparison per frame'}
caps.flags.plot.phi     = 0;
caps.method.thetas      = 'eig';
caps.method.phis        = 'eig';
caps.method.norm.phi     = 0;


for tap = 7:7

    caps.tap = tap;

    for kdata = 1:length(Dat)
    
        caps                    = extract_infos(caps,Dat,kdata);
        [thetas,phis, H_est]    = channel_estimation(Dat,caps);
        [thetas, phis]          = extract_ground_truth(Dat,caps,thetas,phis);
        
        if caps.flags.plot.phi
            metrics             = get_metrics(caps,thetas,phis);
            plot_results(caps,H_est, thetas,metrics,phis);
        else
            metrics             = get_metrics(caps,thetas);
            plot_results(caps,H_est, thetas,metrics);
        end


    end
end

cd ../err
Mthetas             = [Carac,metrics.thetas.ErrMean,metrics.thetas.ErrStd,metrics.thetas.ErrRms];
Mthetas(end+1,:)    = [0,median(metrics.thetas.ErrMean),median(metrics.thetas.ErrStd),median(metrics.thetas.ErrRms)];
writematrix(Mthetas,strcat('<Err Theta>-',caps.filename,'.csv'))

if caps.flags.plot.phi
    Mphis             = [Carac,metrics.phis.ErrMean,metrics.phis.ErrStd,metrics.phis.ErrRms];
    Mphis(end+1,:)    = [0,median(metrics.phis.ErrMean),median(metrics.phis.ErrStd),median(metrics.phis.ErrRms)];
    writematrix(Mphis,strcat('<Err Phi>-',caps.filename,'.csv'))
end
cd(myRootPath)

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

    caps.Fn         = allFilenames;
    caps.PathSave   = allPathnames;
end



%-----------------------------------------------------

function out = get_value_from_filename_in(folderPath,quantity,varargin)

    cd(folderPath{1})
  
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