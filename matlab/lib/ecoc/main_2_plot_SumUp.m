% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.2.0
%   Date            : 2024-04-21
%   ArXivs          : 2024-04-15 (1.0.0)
%                   : 2024-04-17 (1.0.1)
%                   : 2024-04-20 (1.1.0) errors -> Err, cleaning
%                     2024-04-21 (1.2.0) [NEW] PolLaw_Lin
%   Licence         : cc-by-nc-sa
%                     Attribution - Non-Commercial - Share Alike 4.0 International
%
% ----- MAIN IDEA -----
%   See VAE ability to tract the State of Polarisation
%   of a beam propagating into an optical fibre.
%
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

%% MAINTENANCE
rst
default_plots()
addpath '/home/louis/L_Libraries/Matlab/louis library/general'
addpath '/home/louis/L_Libraries/Matlab/louis library/mathematics/'

myInitPath      = pwd();
cd ../data/data-ECOC/

%% DEFINITION OF THE PATH 1/2
Rs              = num2str(64);      % {64,128}
what_carac      = 'PolLaw_Lin';     % {PolLaw_Rwalk,PolLaw_Lin,SNRs}
what_mimo       = 'vae';            % {cma,vae}

get_errors      = 1;
plot_all_curves = 1;
skip_train      = 1;

if strcmpi(what_carac,'SNRs')
    SNRs        = linspace(13,28,16);
    Fpol        = 100;
elseif strcmpi(what_carac,'PolLaw_Rwalk')
    SNRs        = 23;
    Fpol        = 10000;
elseif strcmpi(what_carac,'PolLaw_Lin')
    SNRs        = 23;
    Theta_Start = 20;
    Theta_End   = 45+(45-Theta_Start);
    Slope       = 10;
end


%% LOOP
for k =1:length(SNRs)

    %%% definition of the path 2/2
    if strcmpi(what_carac,'SNRs')
        carac_val   = '';
        carac_val2  = sprintf('SNR%i',SNRs(k));

    elseif strcmpi(what_carac,'PolLaw_Rwalk')
        carac_val       = num2str(Fpol);
        carac_val2      = '';

    elseif strcmpi(what_carac,'PolLaw_Lin')
        carac_val       = strcat(num2str(Theta_Start),'-',num2str(Theta_End));
        carac_val2      = num2str(Slope);

    end
    
    if strcmpi(what_carac,'SNRs')
        path_name_suffix_csv = ...
            strcat('Rs',Rs,'/',what_carac,carac_val,'/',...
                'csv/',what_mimo,'/',carac_val2);

    elseif strcmpi(what_carac,'PolLaw_Rwalk')
        path_name_suffix_csv = ...
            strcat('Rs',Rs,'/',what_carac,'/Fpol',carac_val,'/',...
                'csv/',what_mimo);

    elseif strcmpi(what_carac,'PolLaw_Lin')
        path_name_suffix_csv = ...
            strcat('Rs',Rs,'/',what_carac,'/',carac_val,'/',...
                'csv/','slope',carac_val2);
    end
    
    
    if get_errors
        if strcmpi(what_carac,'SNRs')
            path_name_suffix_errors = ...
                strcat('Rs',Rs,'/',what_carac,carac_val,'/',...
                    'Err/',what_mimo,'/',carac_val2);

        elseif strcmpi(what_carac,'PolLaw_Rwalk')
            path_name_suffix_errors = ...
                strcat('Rs',Rs,'/',what_carac,'/Fpol',carac_val,'/',...
                    'Err/',what_mimo,'/',carac_val2);
            
        elseif strcmpi(what_carac,'PolLaw_Lin')
            path_name_suffix_errors = ...
                strcat('Rs',Rs,'/',what_carac,'/',carac_val,'/',...
                'Err/','slope',carac_val2);
        end

    end
    
    if k > 1
        cd ../data/data-ECOC/
    end

    %%% importing data
    cd(path_name_suffix_csv)
    dataCSV = import_data({'.csv'}); % {'.<ext>'} possible input
    cd(myInitPath)
    cd ../data/data-ECOC/
    
    if get_errors
        cd(path_name_suffix_errors)
        [dataErrs,fnamesErrs]= import_data();
        cd(myInitPath)
    end
    
    %%% regimes to catch (w/wo transient)
    if strcmpi(Rs,'64')
        NtrainCSV   = 20;
        NtrainErr   = NtrainCSV;
    else
        NtrainCSV   = 40+10;
        NtrainErr   = 40;
    end


    %%% data initialisation + import 1/2 - wo 'errors'
    Ndata               = length(dataCSV);
    iterCSV             = 1:length(dataCSV{1}.iteration);
    NiterCSV            = length(iterCSV);
    loss                = zeros(NiterCSV,Ndata);
    SER                 = zeros(NiterCSV,Ndata);
    Thetas              = zeros(NiterCSV,Ndata);
    
    for j = 1:Ndata
        loss(:,j)       = dataCSV{j}.loss;
        SER(:,j)        = dataCSV{j}.SER;
        Thetas(:,j)     = dataCSV{j}.Thetas;
    end
    
    clear dataCSV

    %%% data initialisation + import 2/2 - 'errors'
    if get_errors
        cd ../data/data-ECOC/
        cd(path_name_suffix_errors)

        NfilesErrs      = length(fnamesErrs);
        idx             = zeros(1,NfilesErrs);

        %%% removing <Err tht> files
        for j = 1:NfilesErrs
            if contains(fnamesErrs{j},'<')
                idx(j)  = j;
            end
        end

        fnamesErrs(idx~=0)  = [];
        dataErrs(idx~=0)    = [];
        NfilesErrs          = length(dataErrs);

        iterErr             = 1:length(dataErrs{1}.Var1);
        NiterErr            = length(iterErr);
        errors              = zeros(iterErr(end),Ndata);
        
        for j = 1:Ndata
            errors(:,j)     = dataErrs{j}.Var1;
        end
        clear dataErrs
        clear idx
    end
    
    cd(myInitPath)
    
    %%% statistics
    LossMean        = mean(loss,2);
    SERMean         = mean(SER,2);
    ThetasMean      = mean(Thetas,2);
    
    LossStd         = std(loss,0,2);
    SERStd          = std(SER,0,2);
    ThetasStd       = std(Thetas,0,2);
    
    LossRms         = LossStd./LossMean*100;
    SERRms          = SERStd./SERMean*100;
    ThetasRms       = ThetasStd./ThetasMean*100;
    
    %%% saving
    SAVED_Matrix    = [iterCSV(NtrainCSV+1:end)',LossMean(NtrainCSV+1:end), SERMean(NtrainCSV+1:end), ThetasMean(NtrainCSV+1:end),...
                                           LossStd(NtrainCSV+1:end), SERStd(NtrainCSV+1:end),  ThetasStd(NtrainCSV+1:end),...
                                           LossRms(NtrainCSV+1:end), SERRms(NtrainCSV+1:end),  ThetasRms(NtrainCSV+1:end)];

    colnames        = {'iter',  'lossMean','sersMean','thtsMean',...
                                'lossStd', 'sersStd', 'thtsStd',...
                                'lossRms', 'sersRms', 'thtsRms'};
    if get_errors
        ErrsMean        = mean(errors(NtrainErr+1:end,:),2);
        ErrsStd         = std(errors(NtrainErr+1:end,:),0,2);
        ErrsRms         = ErrsStd./ErrsMean*100;
    
        SAVED_Matrix    = insert_column(SAVED_Matrix,ErrsMean,10);
        SAVED_Matrix    = insert_column(SAVED_Matrix,ErrsStd,11);
        SAVED_Matrix    = insert_column(SAVED_Matrix,ErrsRms,12);

        colnames        = [colnames,{'errsMean'},{'errsStd'},{'errsRms'}];
    end
    
    %%% plotting

    if str2double(Rs) == 64
        iter_max_train = 20;
    elseif str2double(Rs) == 128
        iter_max_train = 40;
    end

    f       = figure;
    XLIM    = [min([iterCSV,iterErr]),min([max(iterCSV),max(iterErr)])];

    if ~get_errors
        subplot(1,3,1)
            hold on
            if strcmpi(what_mimo,'vae')
                YLIM = [-1,1]*2000;
            else
                YLIM = [0,3];
            end

            area([0, iter_max_train], [YLIM(1),YLIM(1)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            area([0, iter_max_train], [YLIM(2),YLIM(2)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            [leg,icn] = legend('training');
            set(leg,'box', 'off', 'location', 'southwest')
            set(icn(1),'rotation',90,'Position',[0.1,0.5,0])
            set(icn(2),'visible','off')
            
            if plot_all_curves
                plot(iterCSV,loss,'--k','HandleVisibility','off')
            end

            plot(iterCSV(NtrainCSV+1:end),LossMean(NtrainCSV+1:end),'-k','LineWidth',5,'HandleVisibility','off')
            xlabel('iteration')
            ylabel("loss function")
            ylim(YLIM)
            xlim(XLIM)

        subplot(1,3,2)
            hold on
            if strcmpi(what_mimo,'vae')
                YLIM = [1e-3,1];
            else
                YLIM = [1e-1,1];
            end
            area([0, iter_max_train], [YLIM(1),YLIM(1)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            area([0, iter_max_train], [YLIM(2),YLIM(2)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            [leg,icn] = legend('training');
            set(leg,'box', 'off', 'location', 'southwest')
            set(icn(1),'rotation',90,'Position',[0.1,0.5,0])
            set(icn(2),'visible','off')
                        
            if plot_all_curves
                plot(iterCSV,SER,'--k','HandleVisibility','off')
            end

            plot(iterCSV(NtrainCSV+1:end),SERMean(NtrainCSV+1:end),'-k','LineWidth',5,'HandleVisibility','off')
            set(gca,'Yscale','log')
            xlabel('iteration')
            ylabel('Symbol Error Rate')
            ylim(YLIM)
            xlim(XLIM)
            title(['Rs', Rs,' ',what_mimo, ' ', carac_val2])
    
        subplot(1,3,3)
            hold on
            YLIM = [-1,1]*5*10^(log10(Fpol)-7);
            area([0, iter_max_train], [YLIM(1),YLIM(1)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            area([0, iter_max_train], [YLIM(2),YLIM(2)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            [leg,icn] = legend('training');
            set(leg,'box', 'off', 'location', 'southwest')
            set(icn(1),'rotation',90,'Position',[0.1,0.5,0])
            set(icn(2),'visible','off')
                        
            if plot_all_curves
                plot(iterCSV,Thetas,'--k','HandleVisibility','off')
            end
            plot(iterCSV(NtrainCSV+1:end),ThetasMean(NtrainCSV+1:end),'-k','LineWidth',5,'HandleVisibility','off')
            xlabel('Symbol Error Rate')
            xlabel('iteration')
            ylabel('$\Theta$')
            ylim(YLIM)
            xlim(XLIM)

    elseif get_errors
        subplot(2,2,1)
            hold on

            if strcmpi(what_mimo,'vae')
                YLIM = [-1,1]*2000;
            else
                YLIM = [0,3];
            end

            area([0, iter_max_train], [YLIM(1),YLIM(1)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            area([0, iter_max_train], [YLIM(2),YLIM(2)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            [leg,icn] = legend('training');
            set(leg,'box', 'off', 'location', 'southwest')
            set(icn(1),'rotation',90,'Position',[0.1,0.5,0])
            set(icn(2),'visible','off')
            
            if plot_all_curves
                plot(iterCSV,loss,'--k','HandleVisibility','off')
            end

            plot(iterCSV(NtrainCSV+1:end),LossMean(NtrainCSV+1:end),'-k','LineWidth',5,'HandleVisibility','off')
            xlabel('iteration')
            ylabel("loss function")
            ylim(YLIM)
            xlim(XLIM)

        subplot(2,2,2)
            hold on
            if strcmpi(what_mimo,'vae')
                YLIM = [1e-3,1];
            else
                YLIM = [1e-1,1];
            end

            area([0, iter_max_train], [YLIM(1),YLIM(1)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            area([0, iter_max_train], [YLIM(2),YLIM(2)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            [leg,icn] = legend('training');
            set(leg,'box', 'off', 'location', 'southwest')
            set(icn(1),'rotation',90,'Position',[0.1,0.5,0])
            set(icn(2),'visible','off')

            if plot_all_curves
                plot(iterCSV,SER,'--k','HandleVisibility','off')
            end

            plot(iterCSV(NtrainCSV+1:end),SERMean(NtrainCSV+1:end),'-k','LineWidth',5,'HandleVisibility','off')
            set(gca,'Yscale','log')
            xlabel('iteration')
            ylabel('Symbol Error Rate')
            ylim(YLIM)
            xlim(XLIM)
            title(['Rs', Rs,' ',what_mimo, ' ', carac_val2])
    
        subplot(2,2,3)
            hold on
            if strcmpi(what_carac,'PolLaw_Rwalk')
                YLIM = [-1,1]*5*10^(log10(Fpol)-7);
            end
            area([0, iter_max_train], [YLIM(1),YLIM(1)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            area([0, iter_max_train], [YLIM(2),YLIM(2)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            [leg,icn] = legend('training');
            set(leg,'box', 'off', 'location', 'southwest')
            set(icn(1),'rotation',90,'Position',[0.1,0.5,0])
            set(icn(2),'visible','off')

            if plot_all_curves
               plot(iterCSV,Thetas,'--k','HandleVisibility','off')
            end
            
            plot(iterCSV(NtrainCSV+1:end),ThetasMean(NtrainCSV+1:end),'-k','LineWidth',5,'HandleVisibility','off')
            xlabel('Symbol Error Rate')
            xlabel('iteration')
            ylabel('$\Theta$ [deg]')
            if strcmpi(what_carac,'PolLaw_Rwalk')
                ylim(YLIM)
            end
            xlim(XLIM)

        subplot(2,2,4)
            if ~strcmpi(what_carac,"PolLaw_Lin")
                if strcmpi(what_mimo,"cma")
                    YLIM = [0,25];
                else
                    YLIM = [0,5];
                end
            else
                YLIM = [0,25];
            end
            hold on
            area([0, iter_max_train], [YLIM(1),YLIM(1)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            area([0, iter_max_train], [YLIM(2),YLIM(2)], 'FaceColor', ones(1,3)*0.8, 'EdgeColor', 'none',FaceAlpha=0.5);
            [leg,icn] = legend('training');
            set(leg,'box', 'off', 'location', 'southwest')
            set(icn(1),'rotation',90,'Position',[0.1,0.5,0])
            set(icn(2),'visible','off')
            
            if plot_all_curves
                plot(iterErr,errors,'--k','HandleVisibility','off')
            end

            plot(iterErr(NtrainErr+1:end),abs(ErrsMean),'-k','LineWidth',5,'HandleVisibility','off')
            xlabel('iteration')
            ylabel('$|\Delta\Theta|$ [deg]')
            ylim(YLIM)
            xlim(XLIM)

    end
    
    
    tmp = myInitPath(1:end-6);
    cd(strcat(tmp,'data/data-ECOC/',path_name_suffix_csv))
    if ~strcmpi(what_carac,'PolLaw_Lin')
        cd(strcat('../../../../SumUp/',what_carac,'/',what_mimo))
    else
        cd(strcat('../../../../SumUp/',what_carac))
    end
    
    


    fname_saved     = strcat('SUM_UP___',path_name_suffix_csv);
    fname_saved     = replace(fname_saved,'/','_');
    fname_saved     = replace(fname_saved,'_csv','');
    myTable         = array2table(SAVED_Matrix,"VariableNames",colnames);

    writetable(myTable,[fname_saved,'.csv'])
    saveas(f,string([fname_saved, '.svg']))
    
    if length(SNRs)>1
        close all
    end
    cd(myInitPath)
end






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   default_plots               
%   import_data                 acceptedFormats                 [allData, allFilenames, allPathnames]
%   get_value_from_filename_in  folderPath,quantity,varargin    out
%   get_number_from_string_in   stringIn,what,varargin          out
% ---------------------------------------------

function default_plots()
    set(groot,'defaultAxesTickLabelInterpreter','latex'); 
    set(groot,'defaulttextinterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');
    set(groot,'defaultFigureUnits','normalized')
    set(groot,'defaultFigurePosition',[0 0 1 1])
    set(groot,'defaultAxesFontSize', 18);
    set(groot,'defaultfigurecolor',[1 1 1])
    set(groot,'defaultAxesFontWeight', 'bold')
end
%-----------------------------------------------------

function [allData, allFilenames, allPathnames] = import_data(acceptedFormats,varargin)
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
    
    elseif nargin == 1
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
end
%-----------------------------------------------------

% function out = get_value_from_filename_in(folderPath,quantity,varargin)
% 
%     cd(folderPath)
%   
%     if nargin == 2
%         nfiles          = length(dir(pwd))-2;
%         folder_struct   = dir(pwd);
%         out             = zeros(nfiles,1);
% 
%         for k=1:nfiles
%             filename    = folder_struct(k+2).name;
%             out(k)      = get_number_from_string_in(filename,quantity);
%         end
% 
%     else
%         nfiles          = length(varargin{1});
%         out             = zeros(nfiles,1);
%         for k=1:nfiles
%             out(k)      = get_number_from_string_in(varargin{1}{k},quantity);
%         end
%     end
% 
%     out = sort(out);
%     
% end
%-----------------------------------------------------

% function out = get_number_from_string_in(stringIn,what,varargin)
% 
%     stringIn    = char(stringIn);
%     iwhat       = strfind(stringIn,what);
% 
%     if nargin == 2
%         iendwhat    = iwhat+length(what);
%         idashes     = strfind(stringIn,'-');
%         [~,itmp]    = max(idashes-iendwhat>0);
%         idashNext   = idashes(itmp);
%         strTmp      = stringIn(iendwhat+1:idashNext-1);
%     else
%         if nargin > 2
%             if iwhat-varargin{1}<1
%                 istart = 1;
%             else
%                 istart = iwhat-varargin{1};
%             end
%             if nargin == 4
%                 if iwhat+varargin{2}>length(stringIn)
%                     iend = length(stringIn);
%                 else
%                     iend = iwhat+varargin{2};
%                 end
%             end
%             strTmp  = stringIn(istart:iend);
%         end
%     end
% 
%     indexes = regexp(strTmp,'[0123456789.]');
%     out     = str2double(strTmp(indexes));
% end
%-----------------------------------------------------


function out = insert_column(in,new_col,N)

    out =  [in(:,1:N),new_col,in(:,N+1:end)];
end
