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


%% MAINTENANCE
rst
format short
caps.log.Date = '24-10-14-16';
caracs0   = '<ErrTh>'; % add '-' otherwise it also considers the <ErrTh>
caracs1   = {'CFO',1};         % {Nplts, dnu}
caracs2   = {'vsop',1};       % {fpol, ThEnd}
caracs3   = {'NSbB',[50,100,150,200,250,300,350,400,450]};
caracs4   = {'SNR_dB',17};
caracs5   = {'Rs',128};
caracs6   = {'','mat'};

table_varnames = {'CFO','vsop','NSbB','SNR_dB','Rs','mean','std','rms'};

caps.save.errs          = 1;
caps.save.mean_errs     = 1;
caps.what_carac         = caracs1{1};
nrea                    = 10;

count = 0;
row         = 1;
for ncarac1 = 1:length(caracs1{2})
    fprintf("%s = %.1f\n",caracs1{1},caracs1{2}(ncarac1))

    for ncarac2 = 1:length(caracs2{2})
        fprintf("\t%s = %.1f\n",caracs2{1},caracs2{2}(ncarac2))
    
        for ncarac3 = 1:length(caracs3{2})
            fprintf("\t\t %s = %.1f\n",caracs3{1},caracs3{2}(ncarac3))

            for ncarac4 = 1:length(caracs4{2})
                fprintf("\t\t\t%s = %.1f\n",caracs4{1},caracs4{2}(ncarac4))

                for ncarac5 = 1:length(caracs5{2})
                    fprintf("\t\t\t\t%s = %.1f\n",caracs5{1},caracs5{2}(ncarac5))

%                     cd(strcat('../python/data-',caps.log.Date,"/err/thetas/"))
                    cd(strcat('../python/data-',caps.log.Date,"/err/sum_up_theta"))
                    caps.log.myInitPath     = pwd();
    
                    caracs         = [...
                                      sprintf("%s %.1f",caracs1{1},caracs1{2}(ncarac1));... % if dnu add space
                                      sprintf("%s %.1f",caracs2{1},caracs2{2}(ncarac2));...  % {%d ThEnd,%.1f fpol}
                                      sprintf("%s %d ",caracs3{1},caracs3{2}(ncarac3));...
                                      sprintf("%s %d",caracs4{1},caracs4{2}(ncarac4));...
                                      sprintf("%s %d",caracs5{1},caracs5{2}(ncarac5));...
                                      sprintf("%s%s",caracs6{1},caracs6{2});...
                                      ];
            
                    [allData,caps]  = import_data({'.csv'},caps,caracs); % {,manual selection}
                    cd(caps.log.myInitPath)
%                     disp(row)
                    Errs(row,:) = allData{1}(end,:);
                    % the end line is the median value of the errors of the previous lines
                    
                    cd(caps.log.myRootPath)
                    row = row+1;
                end % carac5
            end % carac4
        end % carac3
    end % carac2
end % carac1

Errs.Properties.VariableNames = table_varnames;
cd error_estimation_theta/csv
writetable(Errs,caps.log.Fn{1})
% cd ..
% f = figure;
% xlabel('$\mathbf{N_{symb,batch}}$')
% 
% set(gcf, 'Position', [0.0198,0.0009,0.5255,0.8824])
%     yyaxis left
%         plot(Errs.NSbB,abs(Errs.mean),...
%             'color', 'k',...
%             'LineStyle', '-', ...
%             'Marker', 'o', ...
%             'MarkerFaceColor','k',...
%             'MarkerEdgeColor','k',...
%             'MarkerSize',15,...
%             'LineWidth', 2);
%         ylabel('$\mathbf{|<\theta-\hat{\theta}>|}$ [deg]', ...
%             Interpreter='latex', FontWeight="bold")
% 
% 
% 
%     yyaxis right
%         plot(Errs.NSbB,Errs.std,...
%             'color', 'b',...
%             'LineStyle', '-', ...
%             'Marker', '^', ...
%             'MarkerSize',15,...
%             'MarkerFaceColor','b',...
%             'MarkerEdgeColor','b',...
%             'LineWidth', 2);
%     ylabel('$\mathbf{\sigma(\theta-\hat{\theta})}$ [deg]',...
%         FontWeight="bold")
% 
% ax = gca;
% ax.YAxis(1).Color = 'k';
% ax.YAxis(2).Color = 'b';
% 
% tmp_fname   = char(caps.log.Fn{1});
% SNR_str     = num2str(caracs4{2});
% Rs_str      = num2str(caracs5{2});
% lgd_str     = strcat('Rs ',Rs_str, '- SNR ',SNR_str);
% 
% if contains(tmp_fname,'fft')
%     lgd_str = strcat('FD-',lgd_str);
% else
%     lgd_str = strcat('TD-',lgd_str);
% end
% legend(lgd_str,Location="northwest")
% 
% saveas(f,[tmp_fname(1:end-4),'.png'])
% saveas(f,[tmp_fname(1:end-4),'.svg'])
% saveas(f,[tmp_fname(1:end-4),'.fig'])

% 
% % ---------------------------------------------
% % ---------------------------------------------
% % ---------------------------------------------
% dnu             = caracs1{2};
% fpol            = caracs2{2};
% 
% % ---------------------------------------------
% % yfpol_m(:,1)    = Errs(Errs.fpol == 1,:).mean;
% % yfpol_m(:,2)    = Errs(Errs.fpol == 10,:).mean;
% yfpol_m(:,3)    = Errs(Errs.fpol == 100,:).mean;
% 
% % yfpol_s(:,1)    = Errs(Errs.fpol == 1,:).std;
% % yfpol_s(:,2)    = Errs(Errs.fpol == 10,:).std;
% yfpol_s(:,3)    = Errs(Errs.fpol == 100,:).std;
% 
% % ---------------------------------------------
% % ydnu_m(:,1)    = Errs(Errs.dnu == 1,:).mean;
% % ydnu_m(:,2)    = Errs(Errs.dnu == 5,:).mean;
% % ydnu_m(:,3)    = Errs(Errs.dnu == 10,:).mean;
% % ydnu_m(:,4)    = Errs(Errs.dnu == 50,:).mean;
% ydnu_m(:,5)    = Errs(Errs.dnu == 100,:).mean;
% 
% % ydnu_s(:,1)    = Errs(Errs.dnu == 1,:).std;
% % ydnu_s(:,2)    = Errs(Errs.dnu == 5,:).std;
% % ydnu_s(:,3)    = Errs(Errs.dnu == 10,:).std;
% % ydnu_s(:,4)    = Errs(Errs.dnu == 50,:).std;
% ydnu_s(:,5)    = Errs(Errs.dnu == 100,:).std;
% 
% % ---------------------------------------------
% 
% figure
% subplot(1,2,1)
% hold on
% % plot(fpol,yfpol_m(1,:),'-r',    DisplayName="$\Delta\nu = 1$",LineWidth= 2)
% % plot(fpol,yfpol_m(2,:),'--b',   DisplayName="$\Delta\nu = 5$",LineWidth= 2)
% % plot(fpol,yfpol_m(3,:),'-.k',   DisplayName="$\Delta\nu = 10$",LineWidth= 2)
% % plot(fpol,yfpol_m(4,:),'-',     DisplayName="$\Delta\nu = 50$",LineWidth= 3)
% plot(fpol,yfpol_m(5,:),'--m',   DisplayName="$\Delta\nu = 100$",LineWidth= 3)
% set(gca,'Xscale','log','Yscale','log')
% ylim([5e-2,5])
% xlabel('$f_{pol}~[kHz]$')
% ylabel('$<\hat{\theta}-\theta>~[deg]$')
% xticks([1,10,100])
% xticklabels({'1','10','100'});
% hleg = legend('show');
% legend boxoff
% title(hleg,'units in [kHz]')
% grid on
% box on
% 
% subplot(1,2,2)
% hold on
% % plot(dnu,ydnu_m(1,:),'-r', DisplayName="$f_{pol} = 1$",LineWidth= 2)
% % plot(dnu,ydnu_m(2,:),'--b',DisplayName="$f_{pol} = 10$",LineWidth= 2)
% plot(dnu,ydnu_m(3,:),'-.k', DisplayName="$f_{pol} = 100$",LineWidth= 2)
% set(gca,'Xscale','log','Yscale','log')
% ylim([5e-2,5])
% xlabel('$\Delta\nu~[kHz]$')
% ylabel('$<\hat{\theta}-\theta>~[deg]$')
% xticks([1,10,100])
% xticklabels({'1','10','100'});
% hleg = legend('show');
% legend boxoff
% title(hleg,'units in [kHz]')
% grid on
% box on

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


