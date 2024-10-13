% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.0.0
%   Date            : 2024-09-21
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-09-21  (1.0.0)
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

caps.log.Date   = '24-10-13';
caracs1         = {"CFO",[0.1,0.5,1,5,10,]};
% caracs1         = {"dnu",[1,10,100]};
caracs2         = {"vsop",[0.1,1,10]};
caracs2b    = {'Rs',128};
% caracs3         = {"SNR_dB",[10]};
caracs3     = {'NSbB',[50,100,150,200,250,300,350,400,450,500]};

what = "csv";
if strcmpi(what,'mat')
    fprintf(join(["\t,","\t,","\t,","\t,","\t,","\t,","\t,","\t,","\t,","mat\n"]))
else
    fprintf("csv\n")
end

for ncarac1 = 1:length(caracs1{2})
    for ncarac2 = 1:length(caracs2{2})
        for ncarac2b = 1:length(caracs2b{2})
            for ncarac3 = 1:length(caracs3{2})

                cd(strcat('../python/data-',caps.log.Date,sprintf("/%s/",what)))
        
                caracs = [ sprintf("%s %.1f",caracs1{1},caracs1{2}(ncarac1));... % if dnu %d add space, if CFO %.1f
                            sprintf("%s %.1f",caracs2{1},caracs2{2}(ncarac2));... % if %.1f vsop
                            sprintf("%s %d",caracs2b{1},caracs2b{2}(ncarac2b));... % Rs
                            sprintf("%s %d ",caracs3{1},caracs3{2}(ncarac3));... % if  NSbB add space
                        ];
               
                [allData,caps]          = import_data({sprintf('.%s',what)},caps,caracs);
                caps.log.myInitPath     = pwd();

                cd(caps.log.myRootPath)
                if strcmpi(what,'mat')
                    fprintf("%s",join(caracs))
                    fprintf(",\tNfiles %i\n",caps.log.Nfiles)
                else
                    fprintf("%i\n",caps.log.Nfiles)
                end

            end % carac3
        end % carac2b
    end % carac2
end % carac1





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   import_data                     (1.1.1)
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

