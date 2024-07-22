% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.0
%   Date            : 2024-07-22
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-16  (1.0.0)
%   2024-07-19  (1.0.1)
%   2024-07-22  (1.1.0)     [NEW] create_folder, go_to_dir, merge_folders, move_files,
%                               set_caps,trunc_folder_names
%                           import_data: managin empty folders
%
% ----- MAIN IDEA -----
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


rst
caps.log.Date = '24-07-19';

caps.carac      = 'dnu';
caps.what_all   = {'.csv',"betas","phis","thetas","fir","poincare","python",".mat",}; % '.' is mandatory for {csv,mat}

for k = 1:length(caps.what_all)
    caps            = go_to_dir(caps,k,'forward');
    [allData,caps]  = import_data({caps.ext},caps); % {,manual selection}
    caps            = set_caps(caps);
    caps            = create_folders(caps);

    move_files(caps)
    merge_folders(caps)
    trunc_folder_names(caps)
    
    fprintf('%s done\n',caps.what)
    caps            = go_to_dir(caps,k,'backward');
end


cd(caps.log.myRootPath)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   create_folders
%   get_number_from_string_in
%   get_value_from_filename_in
%   got_to_dir                  (1.1.0)
%   import_data                 (1.0.2)
%   merge_folders               (1.1.0)
%   move_files                  (1.1.0)
%   set_caps                    (1.1.0)
%   trunc_folder_names          (1.1.0)
% ---------------------------------------------

function caps = create_folders(caps)

if caps.log.Nfiles ~= 0
    cd(caps.log.myInitPath)
    caps.carac_values    = [];
    
    for k = 1:caps.log.Nfiles
        caps.carac_values = get_value_from_filename_in(caps.log.PathSave,caps.carac,caps.log.Fn).';
    end
    
    caps.carac_values    = unique(caps.carac_values);
    
    if ~isfolder("carac")
        cd ../
        for k = 1:length(caps.carac_values)
            mkdir(strcat(caps.prefix," ",caps.carac," ",num2str(caps.carac_values(k))))
        end
        caps.folder_end = pwd;
    end
end

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

function caps = go_to_dir(caps,k,what)

if strcmpi(what,'forward')
    cd(caps.log.myRootPath)
    tmp         = cellstr(caps.what_all);
    caps.what   = char(tmp(k));

    if contains(caps.what,'mat')
        cd(strcat('../python/data-',caps.log.Date,'/',caps.what(2:end)))
        caps.ext = '.mat';
    elseif contains(caps.what,'csv')
        cd(strcat('../python/data-',caps.log.Date,'/',caps.what(2:end)))
        caps.ext = '.csv';
    else
        switch lower(caps.what)
            case 'fir'
                cd(strcat('../python/data-',caps.log.Date,'/figs/fir'))
                caps.ext = ".png";
            case 'python'
                cd(strcat('../python/data-',caps.log.Date,'/figs/python'))
                caps.ext = ".svg";
            case 'poincare'
                cd(strcat('../python/data-',caps.log.Date,'/figs/poincare'))
                caps.ext = ".png";
            case 'betas'
                cd(strcat('../python/data-',caps.log.Date,'/err/betas'))
                caps.ext = ".csv";
            case 'phis'
                cd(strcat('../python/data-',caps.log.Date,'/err/phis'))
                caps.ext = ".csv";
            case 'thetas'
                cd(strcat('../python/data-',caps.log.Date,'/err/thetas'))
                caps.ext = ".csv";
            otherwise
                assert(1==0,'files do not exist')
        end
    end

    caps.ext = char(caps.ext);
else
    index               = strfind(caps.log.myInitPath,caps.log.Date)+2+1+2+1+2; %e.g. 24-07-19
    Nchar               = length(caps.log.myInitPath)-index;
    caps.log.myInitPath = caps.log.myInitPath(1:end-Nchar);
end

end
%-----------------------------------------------------

function [allData, caps] = import_data(acceptedFormats,caps,varargin)
    if nargin < 1
        acceptedFormats = {};
    else
        assert(nargin >= 2, 'not enough arguments')
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
    
    for k = 1:Nfiles
        % [1] file path construction
        selectedFile = fullfile(pathname, filenames{k});

        % [2] file name extraction from file path
        [~, ~, fileExtension] = fileparts(selectedFile);

        % [3] check if no extension specified, load all files
        if isempty(acceptedFormats)
            data            = readtable(selectedFile);
            allData{k}      = data;
            allFilenames{k} = filenames{k};
            allPathnames{k} = pathname;
            continue;
        else
            allFilenames{k} = filenames{k};
            allPathnames{k} = pathname;
            allData{k}      = NaN;
        end

        % [4] file extension check
        if ~any(char(acceptedFormats) == fileExtension)
%             disp(['File format not supported for file: ' filenames{k}]);
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
%                 disp(['File format not supported for file: ' filenames{k}]);
                continue
        end

        allData{k}      = data;
        allFilenames{k} = filenames{k};
        allPathnames{k} = pathname;
    end

    caps.log.Fn         = allFilenames;
    caps.log.PathSave   = allPathnames;
    caps.log.Nfiles     = length(allData);
end
%-----------------------------------------------------

function merge_folders(caps)
if caps.log.Nfiles ~= 0
    for k = 1:length(caps.carac_values)
        source      = strcat(caps.log.myInitPath," ",caps.carac, " ", num2str(caps.carac_values(k)));
        destination = caps.log.myInitPath;
        movefile(source,destination)
    end
end
end
%-----------------------------------------------------

function move_files(caps)

if caps.log.Nfiles ~= 0
    for k = 1:caps.log.Nfiles
        source      = strcat(caps.log.myInitPath,'/',caps.log.Fn{k});
        tmp         = get_value_from_filename_in(caps.log.PathSave,caps.carac,caps.log.Fn{k});
        tmp         = num2str(tmp);
        destination = strcat(caps.folder_end,'/',caps.prefix," ",caps.carac," ",tmp);
    
        movefile(source,destination)
    end
end
end
%-----------------------------------------------------

function caps = set_caps(caps)

    caps.log.myInitPath = pwd();

    if caps.log.Nfiles ~= 0
        if ~strcmpi(caps.what(1),'.')
            caps.prefix = caps.what;
        else
            caps.prefix = caps.what(2:end);
        end
    end

end
%-----------------------------------------------------

function trunc_folder_names(caps)
if caps.log.Nfiles ~= 0
    for k = 1:length(caps.carac_values)
        source      = strcat(caps.log.myInitPath,'/',caps.prefix," ",caps.carac, " ", num2str(caps.carac_values(k)));
        destination = strcat(caps.log.myInitPath,'/',caps.carac," ",num2str(caps.carac_values(k)));
        movefile(source,destination)
    end
end
end
%-----------------------------------------------------