% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.0.0
%   Date            : 2024-10-23
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-10-23  (1.0.0)
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


%% MAINTENANCE
rst
caracs0   = '<ErrTh>';
caracs1   = {'',"fft"};         % {fft,mat}
caracs2   = {'CFO',1};          % {dnu,CFO}
caracs3   = {'vsop',1};
caracs4   = {'Rs',[64,128]};
caracs5   = {'SNR_dB',[17,25]};

mean_std_both   = "M";      % {M==mean,S==std,B==both}
LQ_HQ_fig       = 'LQ';     % {'LQ','HQ','LQ,HQ'}
cd error_estimation_theta/csv

f = figure(Position=[0.0198,0.0009,0.5255,0.8824]);
box on
grid on
hold all
set(gca, 'LineStyleOrder', {'-', '--', ':', '-.'}, 'NextPlot', 'add');
xlabel('$\mathbf{N_{symb,batch}}$')

line_styles = {'-', '--', ':', '-.'};
markers     = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', 'x'};
lgd_str     = [];
j           = 0;



for ncarac1 = 1:length(caracs1{2})
    fprintf("%s%s\n",caracs1{1},caracs1{2}(1))

    for ncarac2 = 1:length(caracs2{2})
        fprintf("\t%s = %.1f\n",caracs2{1},caracs2{2}(ncarac2))
    
        for ncarac3 = 1:length(caracs3{2})
            fprintf("\t\t%s = %.1f\n",caracs3{1},caracs3{2}(ncarac3))

            for ncarac4 = 1:length(caracs4{2})
                fprintf("\t\t\t%s = %.1f\n",caracs4{1},caracs4{2}(ncarac4))
    
                for ncarac5 = 1:length(caracs5{2})
                    fprintf("\t\t\t\t%s = %.1f\n",caracs5{1},caracs5{2}(ncarac5))
                
                caracs         = [...
                                  sprintf("%s%s",caracs1{1},caracs1{2});... % if dnu add space
                                  sprintf("%s %.1f",caracs2{1},caracs2{2}(ncarac2));...  % {%d ThEnd,%.1f fpol}
                                  sprintf("%s %.1f",caracs3{1},caracs3{2}(ncarac3));...
                                  sprintf("%s %d",caracs4{1},caracs4{2}(ncarac4));...
                                  sprintf("%s %d",caracs5{1},caracs5{2}(ncarac5))];
        
                [allData,caps]  = import_data({'.csv'},caps,caracs); % {,manual selection}
                if caps.log.Nfiles ~= 0
                    j = j+1;
                end
                Errs            = allData{1};
                line_style_idx  = mod(j-1, length(line_styles)) + 1;
                marker_idx      = mod(j-1, length(markers)) + 1;
                lgd_str         = [lgd_str,join(caracs)];

if strcmpi(mean_std_both,'B')
    yyaxis left
    plot(Errs.NSbB,abs(Errs.mean),...
        'color', 'k',...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx},...
        'MarkerFaceColor','k',...
        'MarkerEdgeColor','k',...
        'MarkerSize',15,...
        'LineWidth', 2);
    ylabel('$\mathbf{|<\theta-\hat{\theta}>|}$ [deg]', ...
        Interpreter='latex', FontWeight="bold")
    ylim([0,2])
%     set(gca,'Yscale','log')
elseif strcmpi(mean_std_both,'M')
    plot(Errs.NSbB,abs(Errs.mean),...
        'color', 'k',...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx},...
        'MarkerFaceColor','k',...
        'MarkerEdgeColor','k',...
        'MarkerSize',15,...
        'LineWidth', 2);
    ylabel('$\mathbf{|<\theta-\hat{\theta}>|}$ [deg]', ...
        Interpreter='latex', FontWeight="bold")
    ylim([0,2])
%     set(gca,'Yscale','log')
end

if strcmpi(mean_std_both,'B')
    yyaxis right
    plot(Errs.NSbB,Errs.std,...
        'color', 'b',...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx},...
        'MarkerSize',15,...
        'MarkerFaceColor','b',...
        'MarkerEdgeColor','b',...
        'LineWidth', 2);
    ylabel('$\mathbf{\sigma(\theta-\hat{\theta})}$ [deg]',...
        FontWeight="bold")
    set(gca,'Yscale','log')
    ylim([5e-2,10])
elseif strcmpi(mean_std_both,'S')
    plot(Errs.NSbB,Errs.std,...
        'color', 'k',...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx},...
        'MarkerSize',15,...
        'MarkerFaceColor','k',...
        'MarkerEdgeColor','k',...
        'LineWidth', 2);
    ylabel('$\mathbf{\sigma(\theta-\hat{\theta})}$ [deg]',...
        FontWeight="bold")
    set(gca,'Yscale','log')
    ylim([5e-2,10])
end

                end
            end
        end
    end
end

if strcmpi(mean_std_both,'both')
    ax = gca;
    ax.YAxis(1).Color = 'k';
    ax.YAxis(2).Color = 'b';
end

lgd_str = unique(lgd_str);
lgd_str = strrep(lgd_str,'_dB','');
if sum(contains(lgd_str,'fft')) ~= 0
    lgd_str = strrep(lgd_str,'fft','FD-');
elseif sum(contains(lgd_str,'mat')) ~= 0
    lgd_str = strrep(lgd_str,'mat','TD-');
end

lgd = legend(lgd_str);
set(lgd,"location","south outside","NumColumns",2)
legend boxoff
cd ../figs
filename = char(caps.log.Fn{1});
filename = strrep(filename,'NSbB 450 -','');
filename = char(strcat(mean_std_both,'-',filename));
saveas(f,[filename(1:end-3),'fig'])

if contains(LQ_HQ_fig,'HQ')
    saveas(f,[filename(1:end-3),'svg'])
end
if contains(LQ_HQ_fig,'LQ')
    saveas(f,[filename(1:end-3),'png'])
end
cd(caps.log.myRootPath)


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

