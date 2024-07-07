%%
% ---------------------------------------------
% ----- INFORMATIONS -----
%   Function name   : processing_4_errors_only
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   ArXivs          : 
%   Date            : 2024-04-19
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
% cd ../data/data-ECOC/Rs64/PolLaw_Rwalk/Fpol10000/mat/vae/
% cd ../data/data-ECOC/Rs64/SNRs/mat/vae/SNR28
% cd ../data/data-ECOC/Rs64/PolLaw_Rwalk/Fpol100/mat/cma/
% cd '../python/data-2024-04-18/128QAM/17 040/15-75/errors/'
cd ../data/data-ECOC/Rs64/PolLaw_Lin/5-85/Err/slope5

% [Dat,Fn,PathSave]   = import_data({'.csv'},'manual selection');
[Dat,Fn,PathSave]   = import_data({'.csv'});
Nfiles              = length(Dat);
myPath              = PathSave{1};
what_carac          = 'End';  % {dnu, Slope, End}
Carac               = get_value_from_filename_in(myPath,what_carac,Fn);
cd(myInitPath)

Nlines_skip         = sum(Dat{1}.Var1 == 0);
Nerrors             = length(Dat{1}.Var1);
errors              = zeros(Nerrors-Nlines_skip,Nfiles);

for k = 1:Nfiles
    errors(:,k)     = abs(Dat{k}.Var1(Nlines_skip+1:end));
end


ErrMean     = mean(errors);
ErrStd      = std(errors);
ErrRms      = ErrStd./ErrMean*100;

cd(PathSave{1})

M           = [Carac,ErrMean',ErrStd',ErrRms'];
M(end+1,:)  = [0,mean(ErrMean),mean(ErrStd),mean(ErrRms)];
filename    = Fn{1};

cd ../../..
tmp = char(filename);
writematrix(M,strcat('<Err Theta>-',tmp(11:end),'.csv'))
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

