%%
% ---------------------------------------------
% ----- INFORMATIONS -----
%   Function name   : post_processing
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   ArXivs          : 
%   Date            : 2024-04-15 (1.0.0)
%   Version         : 1.0.0
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
%   Author              :
%   Author contact      :
%   Date                :
%   Title of program    :
%   Code version        :
%   Type                : 
%   Web Address         : 
% ----------------------------------------------
%%

%% MAINTENANCE
rst
default_plots()
addpath '/home/louis/L_Libraries/Matlab/louis library/general'
addpath '/home/louis/L_Libraries/Matlab/louis library/mathematics/'

myInitPath      = pwd();
cd ../data/data-ECOC/

Rs          = num2str(64);  % {64,128}
what_carac  = 'SNRs';       % {Fpol,SNRs}
what_mimo   = 'cma';        % {cma,mimo}

if strcmpi(what_carac,'Fpol')
    carac_val = num2str(10000);
else
    carac_val = '';
end

path_name_suffix_csv = ...
    strcat(...
        'Rs',Rs,'/',...
        what_carac,carac_val,'/',...
        'csv/',...
        what_mimo,'/'...
    );

path_name_suffix_errors = ...
    strcat(...
        'Rs',Rs,'/',...
        what_carac,carac_val,'/',...
        'errors/',...
        what_mimo,'/'...
        );

SNRsTh  = linspace(13,28,16);

cd(path_name_suffix_csv)
[dataCSV,fnamesCSV,pnamesCSV]       = import_data(); % {'.<ext>'} possible input

NSNR    = 16;
Ndata   = length(dataCSV);
Nmean   = 25;


for k = 1:Ndata
    SNR         = mean(dataCSV{k}.SNR(end-Nmean:end));
    [~,k_err]    = min(abs(SNR-SNRsTh).^2);
    movefile(strcat(pnamesCSV{k},'/',fnamesCSV{k}),strcat(pnamesCSV{k},sprintf("/SNR%i",SNRsTh(k_err))))
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   default_plots               
%   import_data                 acceptedFormats                 [allData, allFilenames, allPathnames]
%   get_value_from_filename_in  folderPath,quantity,varargin    out
%   get_number_from_string_in   stringIn,what,varargin          out
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

function [allData, allFilenames, allPathnames] = import_data(acceptedFormats)
    if nargin < 1
        acceptedFormats = {};
    end

    if ~iscell(acceptedFormats)
        error("'acceptedFormats' must be a cell array.");
    end

    acceptedFormats = string(acceptedFormats);
    allData         = cell(0);
    allFilenames    = cell(0);
    allPathnames    = cell(0);

    % Automatically select all files in current directory if no formats specified
    if isempty(acceptedFormats)
        pathname = pwd;  % Get current directory
        filenames = dir(fullfile(pathname, '*.*'));  % Get all files
        filenames = {filenames.name};  % Extract file names
        filenames = filenames(~startsWith(filenames, '.'));  % Exclude hidden files
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

    for i = 1:Nfiles
        % [1] file path construction
        selectedFile = fullfile(pathname, filenames{i});

        % [2] file name extraction from file path
        [~, ~, fileExtension] = fileparts(selectedFile);

        % [3] check if no extension specified, load all files
        if isempty(acceptedFormats)
            data = readtable(selectedFile);
            allData{end+1} = data;
            allFilenames{end+1} = filenames{i};
            allPathnames{end+1} = pathname;
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

        allData{end+1}      = data;
        allFilenames{end+1} = filenames{i};
        allPathnames{end+1} = pathname;
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
