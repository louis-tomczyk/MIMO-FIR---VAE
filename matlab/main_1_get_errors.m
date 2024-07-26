% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.2.0
%   Date            : 2024-07-22
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-19  (1.0.0)
%   2024-07-22  (1.0.1)
%   2024-07-23  (1.2.0)     [NEW] go_to_folder, set_caps,
%                           -> standardising matlab codes as in main_1_sort_custom_carac
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


caracs0     = '<Err';
caracs1     = {'dnu',[1,5,10,50,100]};
caracs2     = {'fpol',[1,10,100]};
caracs4     = 'fft';

NrowsErrs   = length(caracs1{1})*length(caracs1{2});

Errs        = table('Size', [NrowsErrs,2+3], ... % 2 = ncaracs, 3 = {mean,std,rms}
                    'VariableTypes', {'double', 'double', 'double', 'double', 'double'}, ...
                    'VariableNames', {caracs1{1}, caracs2{1}, 'mean', 'std', 'rms'});

row         = 1;
for ncarac1 = 1:length(caracs1{2})
    for ncarac2 = 1:length(caracs2{2})

        cd(strcat('../python/data-',caps.log.Date,"/err/thetas"))

        selected_caracs         = [ caracs0;...
                                    sprintf("%s %.0f ",caracs1{1},caracs1{2}(ncarac1));...
                                    sprintf("%s %.1f",caracs2{1},caracs2{2}(ncarac2));...
                                    caracs4];
        [allData,caps]          = import_data({'.csv'},caps,selected_caracs);
        caps.log.myInitPath     = pwd();
        
        disp(row)
        Errs(row,:) = allData{1}(end,:);
        
        cd(caps.log.myRootPath)
        row = row+1;

    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   get_errors
%   get_number_from_string_in
%   get_value_from_filename_in
%   go_to_dir                       (1.2.0)
%   import_data                     (1.0.2)
%   select_files
%   set_caps
% ---------------------------------------------

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

function caps = go_to_dir(caps,what,varargin)

cd(caps.log.myRootPath)

if strcmpi(what,'forward')
    k           = varargin{1};
    tmp         = cellstr(caps.what_all);
    caps.what   = char(tmp(k));
    carac       = varargin{2};

    switch lower(caps.what)
        case 'betas'
            dir_path = strcat('../python/data-',caps.log.Date,'/err/betas');
            caps.ext = ".csv";
        case 'phis'
            dir_path = strcat('../python/data-',caps.log.Date,'/err/phis');
            caps.ext = ".csv";
        case 'thetas'
            dir_path = strcat('../python/data-',caps.log.Date,'/err/thetas');
            caps.ext = ".csv";
        otherwise
            assert(1==0,'files do not exist')
    end

    cd(dir_path)
    filenames   = {dir(fullfile(pwd, '*.*')).name};           % Get all files
    if length(filenames) == 2
        cd(strcat(...
            sprintf('%s',caps.carac.(carac).what), " ", num2str(caps.carac.(carac).value(caps.jphys))...
            ))
    end
end
end
%-----------------------------------------------------

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


function caps = select_files(caps,carac)
    
if caps.log.Nfiles ~= 0
    wanted      = caps.carac.(carac).("what");
    NSbB        = get_value_from_filename_in(caps.log.PathSave,wanted,caps.log.Fn).';
    Ndraws      = sum(NSbB == caps.carac.(carac).("value")(1));
    Fn_2keep    = cell(1,caps.log.Nfiles);
    
    n_files     = 0;
    indexes     = zeros(1,caps.log.Nfiles);
    for k = 1:caps.log.Nfiles
        carac_value_tmp = get_value_from_filename_in(caps.log.PathSave,wanted,caps.log.Fn{k});
        if carac_value_tmp == caps.carac.(carac).("value")(caps.kalgo)
            indexes(k)  = k;
            n_files     = n_files+1;
            Fn_2keep{k} = caps.log.Fn{k};
        end
    end
    
    caps.Fn_2keep       = Fn_2keep(~cellfun('isempty',Fn_2keep));

    caps.indexes        = indexes(indexes ~= 0);
    caps.Nfiles_keep    = n_files;
end
end
%-----------------------------------------------------

