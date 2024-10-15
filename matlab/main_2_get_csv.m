% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.1.0
%   Date            : 2024-10-15
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-07-19  (1.0.0)
%   2024-07-27  (1.1.0) restructuration of the code as the main_0 (2.1.0)
%   2024-07-28  (1.1.1) plotting and saving
%   2024-09-07  (1.1.2) adding 3rd carac
%   2024-10-10  (1.1.3) legend
%   2024-10-13  (2.0.0) handling multiple parameters for the figure
%                       selected_caracs -> caracs
%   2024-10-15  (2.1.0) adding SNR in caracs, scales, legend
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
caps.log.Date = '24-10-14';

% caracs1     = {'NSbB',[50,75,100,125,150,175,200,225,250,300,400,500,750]};
% caracs1     = {'NspT',[13,17,21,25,29]};
caracs1     = {'CFO',[0.1,0.5,1,5,10]};
% caracs1     = {'CFO',[0.1]};
caracs2     = {'vsop',1};
caracs2b    = {'Rs',128};
% caracs3     = {'ma g',[1,5,10,15,20,25]};
% caracs3     = {'SNR_dB',linspace(8,25,18)};
caracs3     = {'NSbB',[500,400,300,250,200,150,100,50]};
entropy     = 5.72;
SNR         = 25;

fec_limit   = 2.8e-2;
time_est    = 0;
% 7 = 3 caracs + ser + time conv + time frame + frame conv

Ntasks  = length(caracs1{2})*length(caracs2{2})*length(caracs2b{2})*length(caracs3{2});
count   = 0;
countT  = 0;

res = struct();
fn_keep = strings(1,length(caracs1{2})*length(caracs2{2})*length(caracs2b{2}));

for ncarac1 = 1:length(caracs1{2})
    fprintf("\n\t %s = %.1f\n",caracs1{1},caracs1{2}(ncarac1))
    for ncarac2 = 1:length(caracs2{2})
        fprintf("\t\t %s = %.1f\n",caracs2{1},caracs2{2}(ncarac2))
        for ncarac2b = 1:length(caracs2b{2})
            fprintf("\t\t %s = %.1f\n",caracs2b{1},caracs2b{2}(ncarac2b))

            for ncarac3 = 1:length(caracs3{2})
                cd(strcat('../python/data-',caps.log.Date,"/csv"))
        
                % change format %d to %.1f is not NSbB nor dnu
                caracs         = [sprintf("%s %.1f",caracs1{1},caracs1{2}(ncarac1));... % if dnu add space, if PhiEnd
                                  sprintf("%s %.1f",caracs2{1},caracs2{2}(ncarac2));... % if vsop %d
                                  sprintf("%s %d",caracs2b{1},caracs2b{2}(ncarac2b));...
                                  sprintf("%s %d ",caracs3{1},caracs3{2}(ncarac3));...
                                  sprintf("SNR_dB %d ",SNR);...
                                            ];
    
                [allData,caps]          = import_data({'.csv'},caps,caracs);
                caps.log.myInitPath     = pwd();
                
                if caps.log.Nfiles ~= 0
                    matrix_tmp              = zeros(caps.log.Nfiles,8); % 8 if carac2b, 7 otherwise
                    location                = zeros(1,caps.log.Nfiles);
                    Niter                   = allData{1}.iteration(end);
                    thetas                  = zeros(Niter,caps.log.Nfiles);
                    bers                    = zeros(Niter,caps.log.Nfiles);
                    dt                      = zeros(Niter,caps.log.Nfiles);
            
                    if sum(contains(allData{1}.Properties.VariableNames,'Phis'))
                        phis                = zeros(Niter,caps.log.Nfiles);
                    end

                    for k = 1:caps.log.Nfiles
                        tmp                 = allData{k}.Thetas == 0;
                        FrameChannel        = length(allData{k}.Thetas(tmp));
                        thetas(:,k)         = allData{k}.Thetas;
                        bers(:,k)           = allData{k}.SER/entropy;
                        
                        if time_est
                            dt(:,k)         = allData{k}.dt;
                        end
            
                        if sum(contains(allData{k}.Properties.VariableNames,'Phis'))
                            phis(:,k)       = allData{1}.Phis;
                        end
            
                        % estimation of convergence rate
                        bers_2      = bers(FrameChannel+1:end,k);
                        bers_3      = bers_2(bers_2<=fec_limit);
                        npass       = length(bers_3);
                        rate_conv   = npass/length(bers_2);

                        x1                 = caracs1{2}(ncarac1);
                        x2                 = caracs2{2}(ncarac2);
                        x12                = caracs2b{2}(ncarac2b);
                        x3                 = caracs3{2}(ncarac3);
                        x4                 = mean(bers_2);
                        x4b                = std(bers_2);
                        x5                 = rate_conv;

                        if time_est
                            x6                 = mean(mean(dt));
                        else
                            x6                 = nan;
                        end

                        matrix_tmp(k,:)     = [x1,x2,x12,x3,x4,x4b,x5,x6];
                    end
                    
                    if size(matrix_tmp,1) ~= 1
                        matrix = mean(matrix_tmp);
                    else
                        matrix = matrix_tmp;
                    end
                    cd(caps.log.myRootPath)
                    
                    if ~exist('T','var')
                        T = array2table(matrix,'VariableNames', ...
                            {caracs1{1},caracs2{1},caracs2b{1},caracs3{1},'meanBER','stdBER','RateConv','TIMEframe'});

                    else
                        Ttmp = array2table(matrix,'VariableNames', ...
                            {caracs1{1},caracs2{1},caracs2b{1},caracs3{1},'meanBER','stdBER','RateConv','TIMEframe'});

                        T = [T;Ttmp];
                    end
                    
                else
                    cd(caps.log.myRootPath)
                    continue
                end
                count = count + 1;
                fprintf('Progress: %.1f/100 --- %s - %s - %s - %s - %s\n',...
                    round(count/Ntasks*100,1),...
                    caracs');
            end

            countT = countT +1;
            res.(sprintf('T%d',countT)) = T;

            if caps.log.Nfiles ~= 0
                filename = char(caps.log.Fn{1});
                writetable(T,filename)
                clear T
            else
                continue
            end
            
            fn_keep(1,countT) = filename;
        
        end
    end
end


f = figure;
hold all
xlabel('$\mathbf{N_{Symb,Batch}}$')
grid on


line_styles = {'-', '--', ':', '-.'};
markers     = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', 'x'};


set(gca, 'LineStyleOrder', {'-', '--', ':', '-.'}, 'NextPlot', 'add');
xlim([25,525])

ttRC    = strings(length(fieldnames(res)),3);
ttBER   = strings(length(fieldnames(res)),3);

ttRC(:,1)    = "CR - ";
ttBER(:,1)   = "BER - ";

for j = 1:length(fieldnames(res))

    line_style_idx  = mod(j-1, length(line_styles)) + 1;
    marker_idx      = mod(j-1, length(markers)) + 1;

    T           = res.(sprintf('T%d',j));
    what        = {'vsop','CFO'};
    index       = zeros(1,length(what));
    what_val    = zeros(1,length(what));
    filename    = fn_keep{j};

    for k = 1:length(what)

        index(k)       = findstr(filename,what{k});
        
        if strcmpi(what{k},'lr')
            what_val(k) = str2double(filename(index(k)+length(what{k}):index(k)+6));
            what_val(k) = what_val(k)/1e3;
        else
            what_val(k) = str2double(filename(index(k)+length(what{k}):index(k)+7));
        end
    end

    for k = 1:length(what)
        ttRC(j,k+1)   = sprintf('%s - %.1f',what{k},what_val(k));
        ttBER(j,k+1)  = sprintf('%s - %.1f',what{k},what_val(k));
    end


    yyaxis left
        semilogy(T.NSbB,T.RateConv,...
            'color', 'k',...
            'LineStyle', line_styles{line_style_idx}, ...
            'Marker', markers{marker_idx}, ...
            'MarkerFaceColor','k',...
            'MarkerEdgeColor','k',...
            'MarkerSize',15,...
            'LineWidth', k,...
            DisplayName=join(ttRC(j,:)));
        ylabel('Success Rate (SR)',FontWeight='bold')
    %     ylim([1e-1,1.01])
    %     set(gca,'Yscale','log')

        ylim([0,1.01])


    yyaxis right
        errorbar(T.NSbB,T.meanBER,T.stdBER,...
            'color', 'b',...
            'LineStyle', line_styles{line_style_idx}, ...
            'Marker', markers{marker_idx},...
            'MarkerSize',15,...
            'MarkerFaceColor','b',...
            'MarkerEdgeColor','b',...
            'LineWidth', k,...
            DisplayName=join(ttBER(j,:)));

        ylabel('Bit Error Rate (BER)',FontWeight='bold')
        if contains(caracs{end},'17')
            ylim([1e-2,1])
        else
            ylim([1e-5,1])
        end
        set(gca,'Yscale','log')

end


ax = gca;
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = 'b';

yyaxis right
    h = plot([min(caracs3{2}),max(caracs3{2})],[1,1]*fec_limit, ...
        '-r',LineWidth=5);
    annotation('textbox',...
    [0.56,0.89,0.35,0.04],'Color',[1 0 0],...
    'String',{'$\mathbf{FEC~limit = 2.8\cdot 10^{-2}}$'},...
    'Interpreter','latex',...
    'FontWeight','bold',...
    'FontSize',20,...
    'FitBoxToText','off',...
    'EdgeColor','none');
    set(h,"DisplayName",'')



box on
lgd = legend(Location="southoutside",NumColumns=2);
entries = get(lgd, 'String');
newEntries = entries(1:end-1);
set(lgd, 'String', newEntries);
legend boxoff
set(gcf, 'Position', [0.0198,0.0009,0.5255,0.8824])


if SNR == 17 && caracs2b{2} == 64
    title(lgd,'\textbf{@SNR = 17 [dB], $R_s =$ 64 [GBd], CFO $\times$ 10 [kHz], $v_{SoP} \times$ 10 [krad/s]}');

elseif SNR == 17 && caracs2b{2} == 128
    title(lgd,'\textbf{@SNR = 17 [dB], $R_s =$ 128 [GBd], CFO $\times$ 10 [kHz], $v_{SoP} \times$ 10 [krad/s]}');

elseif SNR == 25 && caracs2b{2} == 64
    title(lgd,'\textbf{@SNR = 25 [dB], $R_s =$ 64 [GBd], CFO $\times$ 10 [kHz], $v_{SoP} \times$ 10 [krad/s]}');

elseif SNR == 25 && caracs2b{2} == 128
    title(lgd,'\textbf{@SNR = 25 [dB], $R_s =$ 128 [GBd], CFO $\times$ 10 [kHz], $v_{SoP} \times$ 10 [krad/s]}');

end




% saveas(f,[filename(1:end-3),'fig'])
% saveas(f,[filename(1:end-3),'svg'])
saveas(f,[filename(1:end-3),'png'])
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

