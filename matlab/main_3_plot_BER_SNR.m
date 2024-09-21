% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.0.1
%   Date            : 2024-09-10
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-09-09  (1.0.0)
%   2024-09-10  (1.0.1) automatic legend
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
[allData,caps]          = import_data({'.csv'},caps);
caps.log.myInitPath     = pwd();



keys            = {'dnu','PhEnd','fpol','ThEnd'};
leg_keys        = strings();
leg_keys_vals   = zeros(caps.log.Nfiles,4);
leg_keys_kept   = strings();

for k = 1:caps.log.Nfiles
    fname = char(caps.log.Fn{k});
    for j = 1:length(keys)
        if contains(fname,keys(j))
            key_len             = length(keys{j});
            index_dashes        = strfind(fname,'-');
            index_key           = strfind(fname,keys{j});
            index_val_min       = index_key+key_len;
            index_val_max       = index_val_min+min(abs(index_dashes-(index_key+key_len)))-1;
            leg_keys_vals(k,j)  = str2double(fname(index_val_min:index_val_max));
            tmp_key             = keys(j);
            tmp_val             = str2double(fname(index_val_min:index_val_max));
            leg_keys(k,j)       = strcat(tmp_key," ",num2str(tmp_val));

            if contains(keys(j), 'fpol')
                leg_keys(k,j)   = strcat(leg_keys(k,j)," [Hz]-");
            elseif contains(keys(j), 'dnu')
                leg_keys(k,j)   = strcat(leg_keys(k,j)," [kHz]-");
            elseif contains(keys(j), 'End')
                leg_keys(k,j)   = strcat(leg_keys(k,j)," [deg]-");
            end

        end
    end
    tmp = char(join(cellstr(leg_keys(k,leg_keys_vals(k,:)~=0))));
    if strcmpi(tmp(end),'-')
        leg_keys_kept(k) = tmp(1:end-1);
    end
    leg_keys_kept(k)
end


SNRmin          = zeros(1,caps.log.Nfiles);
SNRmax          = zeros(1,caps.log.Nfiles);

for k = 1:caps.log.Nfiles
    SNRmin(k) = min(allData{k}.SNR_dB);
    SNRmax(k) = max(allData{k}.SNR_dB);
end

SNRdBs  = max(SNRmin):min(SNRmax);
tmp     = max(SNRmin)-min(SNRmin)+1;

for k = 1:caps.log.Nfiles
    indexesMin(k)   = tmp-(SNRmin(k)-min(SNRmin));
end

colors = lines(10);
line_styles = {'-', '--', ':', '-.'};
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', 'x'};


figure1 = figure;
hold on
set(gca, 'ColorOrder', lines(10), 'LineStyleOrder', {'-', '--', ':', '-.'}, 'NextPlot', 'add');
for k = 1:caps.log.Nfiles
    color_idx = mod(k-1, size(colors, 1)) + 1;
    line_style_idx = mod(k-1, length(line_styles)) + 1;
    marker_idx = mod(k-1, length(markers)) + 1;

    plot(SNRdBs,allData{k}.BER(indexesMin(k):end),...
        'Color', colors(color_idx, :), ...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx}, ...
        'MarkerFaceColor', colors(color_idx, :),...
        'MarkerEdgeColor', colors(color_idx, :),...
        'LineWidth', k*0.5);
end
plot([5,25],[1,1]*2.8e-2,'-r',LineWidth=5)
xlabel("SNR [dB]")
ylabel('BER')
legend(leg_keys_kept, Location="southwest")
set(gca,"YScale",'log')
grid on
axis square
box on
legend boxoff
set(gcf, 'Position', [0.0198,0.0009,0.5255,0.8824])

annotation(figure1,'textbox',...
    [0.22 0.74 0.35 0.04],'Color',[1 0 0],...
    'String',{'FEC limit = 2.8 10^{-2}'},...
    'FontWeight','bold',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor','none');

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

