% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.1
%   Date            : 2024-07-28
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-19  (1.0.0)
%   2024-07-27  (1.1.0) restructuration of the code as the main_0 (2.1.0)
%   2024-07-28  (1.1.1) plotting and saving
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

%% import data
rst
caps.log.Date = '24-07-28';

caracs1     = {'NSbB',[50,75,100,125,150,175,200,225,250,300,400,500,750]};
% caracs1     = {'NSbB',[300,400,500,750]};
caracs2     = {'dnu',[1]};
entropy     = 5.72;


matrix = zeros(length(caracs1{2}),4);

for ncarac1 = 1:length(caracs1{2})
    for ncarac2 = 1:length(caracs2{2})

        cd(strcat('../python/data-',caps.log.Date,"/csv050"))

        % change format %d to %.1f is not NSbB nor dnu
        selected_caracs         = [ sprintf("%s %d ",caracs1{1},caracs1{2}(ncarac1));...
                                    sprintf("%s %d",caracs2{1},caracs2{2}(ncarac2))];

        [allData,caps]          = import_data({'.csv'},caps,selected_caracs);
        caps.log.myInitPath     = pwd();

        Niter       = allData{1}.iteration(end);
        thetas      = zeros(Niter,caps.log.Nfiles);
        sers        = zeros(Niter,caps.log.Nfiles);
        dt          = zeros(Niter,caps.log.Nfiles);

        if sum(contains(allData{1}.Properties.VariableNames,'Phis'))
            phis            = zeros(Niter,caps.log.Nfiles);
        end

        for k = 1:caps.log.Nfiles

            thetas(:,k)     = allData{k}.Thetas;
            sers(:,k)       = allData{k}.SER/entropy;
            dt(:,k)         = allData{k}.dt;

            if sum(contains(allData{k}.Properties.VariableNames,'Phis'))
                phis(:,k)   = allData{1}.Phis;
            end

            sers_2                      = sers(:,k);
            sers_2(allData{1}.SER>5e-2) = 1;
            [~,location]                = max(sers_2 ~= 1, [], 'omitnan');

        end

        x1                  = caracs1{2}(ncarac1);
        x2                  = mean(sers(location:location+5));
        x3                  = mean(dt)*location;
        x4                  = location;
        matrix(ncarac1,:)   = [x1,x2,x3,x4];

        cd(caps.log.myRootPath)

    end

end

T = array2table(matrix,'VariableNames',{'NSbB','SER','TIME','Frame'});


filename = char(caps.log.Fn{1});
filename = erase(filename,strcat(selected_caracs{1}, ' - '));
lr_index = findstr(filename,'lr');
lr       = str2double(filename(lr_index+2:lr_index+6))/1e3;


writetable(T,filename)



f = figure;
colororder({'k','b'})
xlabel('$log(N_{Symb,Batch})$')
grid on
yyaxis left
    semilogx(T.NSbB,T.TIME,'--k',LineWidth=2,marker = 'square', markersize = 10,DisplayName=sprintf("Time - lr = %.2e",lr))
    ylabel('time for convergence [s]')

yyaxis right
    loglog(T.NSbB,T.SER,'--b',LineWidth=2,marker = 'o', markersize = 10,DisplayName=sprintf("BER - lr = %.2e",lr))
    ylabel('Bit Error Rate')

legend()
saveas(f,[filename(1:end-3),'fig'])



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

