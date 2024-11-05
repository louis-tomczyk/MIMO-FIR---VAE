% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.3.0
%   Date            : 2024-11-05
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2023-10-09  (1.0.0)
%   2024-03-04  (1.1.0) [NEW] plot poincare sphere
%   2024-03-29  (1.1.1) data.FrameChannel -> data.FrameChannel
%   2024-04-18  (1.1.3) IMPORT_DATA
%   2024-04-19  (1.1.4) <Err Theta>
%   ------------------
%   2024-07-06  (2.0.0) encapsulation into modules
%                       [REMOVED] check_if_fibre_prop
%   2024-07-09  (2.0.1) phase estimation
%   2024-07-10  (2.0.2) flexibility and naming standardisation
%   2024-07-11  (2.0.3) cleaning caps structure 
%   2024-07-12  (2.0.4) phase noise management --- for rx['mode'] = 'pilots'
%                       IMPORT_DATA: caps structuring
%   2024-07-16  (2.0.5) multiple files processing
%   2024-07-19  (2.0.6) IMPORT_DATA: managing files not containing data
%   2024-07-23  (2.0.7) progression bar
%   2024-07-25  (2.1.0) IMPORT_DATA (1.1.0): finner selection of files
%                       wrong values saved in <Err *> corrected
%   2024-07-26  (2.1.1) improved progression bar and selection of files
%                       IMPORT_DATA (1.1.1): wrong loop conditin for finner
%                           selection of files
%   2024-10-10  (2.1.2) cleaning + carac5
%   2024-10-28  (2.2.0) metrics.phis.ErrMedian(end) -> metrics.phis.ErrMedian(kdata)
%   2024-11-05  (2.3.0) adding AWGN,
%                       IMPORT_DATA (1.1.2) raise error if no file
%
% ----- MAIN IDEA -----
%   See VAE ability to tract the State of Polarisation
%   of a beam propagating into an optical fibre.
%
% ----- INPUTS -----
% ----- BIBLIOGRAPHY -----
%   Functions           :
%   Author              : Diane PRATO
%   Author contact      : diane.prato@telecom-paris.fr
%   Date                : 2023-06
%   Title of program    : plot_H
%   Code version        : 1.0
%   Type                : 
%   Web Address         : 
% ----------------------------------------------
%%

%% MAINTENANCE
rst
format long
caps.log.Date = '24-11-05';
caracs1   = {'NSbF',[10]};         % {Nplts, dnu}
caracs2   = {'ThEnd',0};       % {fpol, ThEnd}
% caracs3   = {'NSbB',[50,100,150,200,250,300,350,400,450,500]};
caracs3   = {'NSbB',[250]};
caracs4   = {'SNR_dB',[25,20]};
caracs5   = {'Rs',[64]};

caps.plot.fir           = 1;
caps.plot.poincare      = 0;
caps.plot.SOP.xlabel    = 'comparison per frame';   % {'error per frame','error per theta''comparison per frame'}
caps.plot.phis.xlabel   = 'comparison per batch';
caps.method.thetas      = 'mat';                    % {fft, mat, svd}
caps.method.phis        = 'eig';
caps.save.errs          = 1;
caps.save.mean_errs     = 1;
caps.what_carac         = caracs1{1};
nrea                    = 1;

count = 0;
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

                    cd(strcat('../python/data-',caps.log.Date,"/mat"))
                    caps.log.myInitPath     = pwd();
    
                    caracs         = [sprintf("%s %.1f",caracs1{1},caracs1{2}(ncarac1));... % if dnu add space
                                      sprintf("%s %d",caracs2{1},caracs2{2}(ncarac2));...  % {%d ThEnd,%.1f fpol}
                                      sprintf("%s %d ",caracs3{1},caracs3{2}(ncarac3));...
                                      sprintf("%s %d",caracs4{1},caracs4{2}(ncarac4));...
                                      sprintf("%s %d",caracs5{1},caracs5{2}(ncarac5))];
            
                    [allData,caps]  = import_data({'.mat'},caps,caracs); % {,manual selection}
                    cd(caps.log.myInitPath)
                        NfilesTot   = length(caracs1{2})*...
                                      length(caracs2{2})*...
                                      length(caracs3{2})*...
                                      length(caracs4{2})*...
                                      length(caracs5{2})*nrea;
                    
                    for kdata = 1:length(allData)
                    
                        data                        = allData{kdata};
                        caps.kdata                  = kdata;
                        caps                        = extract_infos(caps,data);
                        [caps,thetas,phis,H_est]    = channel_estimation(data,caps); % [deg]
                        [thetas, phis]              = extract_ground_truth(data,caps,thetas,phis);
                        
                        if caps.phis_est
                            metrics         = get_metrics(caps,thetas,phis);
                            plot_results(caps,H_est, thetas,metrics,phis);
                        else
                            metrics         = get_metrics(caps,thetas);
                            plot_results(caps,H_est, thetas,metrics);
                        end
                    
                        cd ../err/thetas
                    
                        if kdata == 1
                            Mthetas         =  zeros(caps.log.Nfiles+1,length(caracs)+3);
                            if caps.phis_est
                                Mphis       =  zeros(caps.log.Nfiles+1,length(caracs)+3);
                            end
                        end
    
                        Mthetas(kdata,:)    = [caracs1{2}(ncarac1),...
                                               caracs2{2}(ncarac2),...
                                               caracs3{2}(ncarac3),...
                                               caracs4{2}(ncarac4),...
                                               caracs5{2}(ncarac5),...
                                                    metrics.thetas.ErrMedian,...
                                                    metrics.thetas.ErrStd,...
                                                    metrics.thetas.ErrRms];
                    
                        if caps.phis_est
                            cd ../phis
                            Mphis(kdata,:)  = [caracs1{2}(ncarac1),...
                                               caracs2{2}(ncarac2),...
                                               caracs3{2}(ncarac3),...
                                               caracs4{2}(ncarac4),...
                                               caracs5{2}(ncarac5),...
                                                    metrics.phis.ErrMedian(kdata),...
                                                    metrics.phis.ErrStd(kdata),...
                                                    metrics.phis.ErrRms(kdata)];
            
                        end
                    
            
                        count       = count + 1;
                        fprintf('Progress: %.1f/100 --- %s - %s - %s- %s- %s\n',...
                            round(count/NfilesTot*100,1),...
                            caracs');
                    end
                    
                    
                    if caps.save.mean_errs && caps.log.Nfiles > 0
                        cd ../thetas
                        Mthetas(end,:)  = [caracs1{2}(ncarac1),...
                                           caracs2{2}(ncarac2),...
                                           caracs3{2}(ncarac3),...
                                           caracs4{2}(ncarac4),...
                                           caracs5{2}(ncarac5),...
                                               median(Mthetas(1:end-1,end-2)),...
                                               median(Mthetas(1:end-1,end-1)),...
                                               median(Mthetas(1:end-1,end))];
            
                        str_tmp         = strcat( ...
                                            sprintf('<ErrTh>-%s-',caps.method.thetas), ...
                                            caps.log.filename(11:end),'.csv'); % 11 == 'matlabX - '
            
                        T = array2table(Mthetas,'VariableNames', ...
                            {caracs1{1}, ...
                             caracs2{1}, ...
                             caracs3{1}, ...
                             caracs4{1}, ...
                             caracs5{1} ...
                            'mean','std','rms'});
%                         writetable(single(Mthetas),str_tmp)
                        cd(caps.log.myInitPath)
%                         cd ../err/sum_up_theta
%                             writematrix(real(Mthetas),str_tmp)
%                         cd(caps.log.myInitPath)
%                         cd ../err/thetas

                        if caps.phis_est
                            cd ../phis

                            Mphis(end,:)= [caracs1{2}(ncarac1),...
                                           caracs2{2}(ncarac2),...
                                           caracs3{2}(ncarac3),...
                                           caracs4{2}(ncarac4),...
                                           caracs5{2}(ncarac5),...
                                               median(Mphis(1:end-1,end-2)),...
                                               median(Mphis(1:end-1,end-1)),...
                                               median(Mphis(1:end-1,end))];
                            str_tmp         = strcat( ...
                                                sprintf('<ErrPh>-%s-',caps.method.phis), ...
                                            caps.log.filename(11:end),'.csv'); % 11 == 'matlabX - '
                            writematrix(Mphis,str_tmp)
                        end
                    end

                    cd(caps.log.myRootPath)
                end
            end
    end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----             
%   import_data         (1.1.2)
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

    if Nfiles == 0 && ~isempty(varargin)
        error(sprintf('\n\n\t\t\tno files found for: %s\n\n',join(varargin{1})))
    elseif Nfiles == 0
        error('no files found.')
    end
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

