% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.0.0
%   Date            : 2024-07-26
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-19  (1.0.0)
%   2024-07-22  (1.0.1)
%   2024-07-23  (1.2.0) [NEW] go_to_folder, set_caps,
%                       -> standardising matlab codes as in main_1_sort_custom_carac
%   2024-07-26  (2.0.0) restructuration + plot
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

% ---------------------------------------------
% ---------------------------------------------
% ---------------------------------------------
dnu             = caracs1{2};
fpol            = caracs2{2};

% ---------------------------------------------
yfpol_m(:,1)    = Errs(Errs.fpol == 1,:).mean;
yfpol_m(:,2)    = Errs(Errs.fpol == 10,:).mean;
yfpol_m(:,3)    = Errs(Errs.fpol == 100,:).mean;

yfpol_s(:,1)    = Errs(Errs.fpol == 1,:).std;
yfpol_s(:,2)    = Errs(Errs.fpol == 10,:).std;
yfpol_s(:,3)    = Errs(Errs.fpol == 100,:).std;

% ---------------------------------------------
ydnu_m(:,1)    = Errs(Errs.dnu == 1,:).mean;
ydnu_m(:,2)    = Errs(Errs.dnu == 5,:).mean;
ydnu_m(:,3)    = Errs(Errs.dnu == 10,:).mean;
ydnu_m(:,4)    = Errs(Errs.dnu == 50,:).mean;
ydnu_m(:,5)    = Errs(Errs.dnu == 100,:).mean;

ydnu_s(:,1)    = Errs(Errs.dnu == 1,:).std;
ydnu_s(:,2)    = Errs(Errs.dnu == 5,:).std;
ydnu_s(:,3)    = Errs(Errs.dnu == 10,:).std;
ydnu_s(:,4)    = Errs(Errs.dnu == 50,:).std;
ydnu_s(:,5)    = Errs(Errs.dnu == 100,:).std;

% ---------------------------------------------

figure
subplot(1,2,1)
hold on
plot(fpol,yfpol_m(1,:),'-r',    DisplayName="$\Delta\nu = 1$",LineWidth= 2)
plot(fpol,yfpol_m(2,:),'--b',   DisplayName="$\Delta\nu = 5$",LineWidth= 2)
plot(fpol,yfpol_m(3,:),'-.k',   DisplayName="$\Delta\nu = 10$",LineWidth= 2)
plot(fpol,yfpol_m(4,:),'-',     DisplayName="$\Delta\nu = 50$",LineWidth= 3)
plot(fpol,yfpol_m(5,:),'--m',   DisplayName="$\Delta\nu = 100$",LineWidth= 3)
set(gca,'Xscale','log','Yscale','log')
ylim([5e-2,5])
xlabel('$f_{pol}~[kHz]$')
ylabel('$<\hat{\theta}-\theta>~[deg]$')
xticks([1,10,100])
xticklabels({'1','10','100'});
hleg = legend('show');
legend boxoff
title(hleg,'units in [kHz]')
grid on
box on

subplot(1,2,2)
hold on
plot(dnu,ydnu_m(1,:),'-r', DisplayName="$f_{pol} = 1$",LineWidth= 2)
plot(dnu,ydnu_m(2,:),'--b',DisplayName="$f_{pol} = 10$",LineWidth= 2)
plot(dnu,ydnu_m(3,:),'-.k', DisplayName="$f_{pol} = 100$",LineWidth= 2)
set(gca,'Xscale','log','Yscale','log')
ylim([5e-2,5])
xlabel('$\Delta\nu~[kHz]$')
ylabel('$<\hat{\theta}-\theta>~[deg]$')
xticks([1,10,100])
xticklabels({'1','10','100'});
hleg = legend('show');
legend boxoff
title(hleg,'units in [kHz]')
grid on
box on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   get_number_from_string_in
%   get_value_from_filename_in
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


