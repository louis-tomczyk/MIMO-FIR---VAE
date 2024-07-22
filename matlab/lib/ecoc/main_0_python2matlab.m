%%
% ---------------------------------------------
% ----- INFORMATIONS -----
%   Function name   : processing_0_python2matlab
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   ArXivs          : 2023-10-09 (1.0.0)
%                   : 2024-03-04 (1.1.0) [NEW] plot poincare sphere
%                   : 2024-03-29 (1.1.1) FrameChannel -> FrameChannel
%                   : 2024-04-18 (1.1.3) import_data
%   Date            : 2024-04-19 (1.1.4) <Err Theta>
%   Version         : 1.1.4
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
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
clear
close all
clc

default_plots()
addpath ./lib


myInitPath      = pwd();
cd ../python/

[Dat,Fn,PathSave]   = import_data({'.mat'},'manual selection');
% [Dat,Fn,PathSave]   = import_data({'.mat'});
Nfiles              = length(Dat);
ErrMean             = zeros(Nfiles,1);
ErrStd              = zeros(Nfiles,1);
ErrRms              = zeros(Nfiles,1);
myPath              = PathSave{1};
what_carac          = 'End';  % {dnu, Slope, End,std}
Carac               = get_value_from_filename_in(myPath,what_carac,Fn);
cd(myInitPath)



plot_flag   = 1;
plot_flag_poincare   = 1;

% plot_SOP    = 'error per theta';   %{'error per frame','error per theta''comparison per frame'}
% plot_SOP    = 'error per frame';   %{'error per frame','error per theta''comparison per frame'}
plot_SOP    = 'comparison per frame';   %{'error per frame','error per theta''comparison per frame'}
%% PROCESSING
for kdata = 1:length(Dat)

disp(kdata)

data            = Dat{kdata};
filename        = char(Fn{kdata});
filename        = ['matlab',filename(1:end-4)];

%%
% ----------------------------------------------
% CHANNEL: ESTIMATION
% ----------------------------------------------
Nframes     = double(data.Nframes);
FrameChannel= double(data.FrameChannel);
frames      = linspace(1,Nframes-FrameChannel,Nframes-FrameChannel);
FIRlength   = double(data.NtapsTX);

H_est       = zeros(2,2,FIRlength);
thetas_est  = zeros(1,Nframes-FrameChannel-1);
H_est_f     = zeros([Nframes-FrameChannel,size(H_est)]);
% Sest        = zeros(3,2,Nframes-FrameChannel);
PDLest      = zeros(Nframes-FrameChannel,1);

for k = 1:Nframes

    h_est_11_I_j        = reshape(data.h_est_liste(k,1,1,1,:),[1,FIRlength]);
    h_est_12_I_j        = reshape(data.h_est_liste(k,1,2,1,:),[1,FIRlength]);
    h_est_21_I_j        = reshape(data.h_est_liste(k,2,1,1,:),[1,FIRlength]);
    h_est_22_I_j        = reshape(data.h_est_liste(k,2,2,1,:),[1,FIRlength]);

    h_est_11_Q_j        = reshape(data.h_est_liste(k,1,1,2,:),[1,FIRlength]);
    h_est_12_Q_j        = reshape(data.h_est_liste(k,1,2,2,:),[1,FIRlength]);
    h_est_21_Q_j        = reshape(data.h_est_liste(k,2,1,2,:),[1,FIRlength]);
    h_est_22_Q_j        = reshape(data.h_est_liste(k,2,2,2,:),[1,FIRlength]);

    H_est(1,1,:)        = complex(h_est_11_I_j, h_est_11_Q_j);
    H_est(1,2,:)        = complex(h_est_12_I_j, h_est_12_Q_j);
    H_est(2,1,:)        = complex(h_est_21_I_j, h_est_21_Q_j);
    H_est(2,2,:)        = complex(h_est_22_I_j, h_est_22_Q_j);

    H_est_f(k,1,1,:)    = fft(H_est(1,1,:));
    H_est_f(k,1,2,:)    = fft(H_est(1,2,:));
    H_est_f(k,2,1,:)    = fft(H_est(2,1,:));
    H_est_f(k,2,2,:)    = fft(H_est(2,2,:));
    H_f0                = squeeze(H_est_f(k,:,:,1));

    thetas_est(k)       = atan(abs(H_f0(1,2)./H_f0(1,1)))*180/pi;   % [deg]
end
thetas_est = thetas_est(FrameChannel+1:end);   % [deg]

FIRest.HH = squeeze(H_est_f(FrameChannel+1:end,1,1,:));
FIRest.HV = squeeze(H_est_f(FrameChannel+1:end,1,2,:));
FIRest.VH = squeeze(H_est_f(FrameChannel+1:end,2,1,:));
FIRest.VV = squeeze(H_est_f(FrameChannel+1:end,2,2,:));

params = {"marker"  ,'o'        ,...
          "size"    , 50        ,...
          "fill"    , 'filled'};

if plot_flag_poincare
    [Sest,f]   = FIR2Stockes(FIRest,params);
end

%%
% ----------------------------------------------
% CHANNEL: GROUND TRUTH
% ----------------------------------------------

h_gnd = struct();
for k = 1:Nframes
    tmp = data.h_channel_liste(k,:,:);
    h_gnd.(sprintf('frame%i',k)).('h11') = tmp(1);
    h_gnd.(sprintf('frame%i',k)).('h12') = tmp(2);
    h_gnd.(sprintf('frame%i',k)).('h21') = tmp(3);
    h_gnd.(sprintf('frame%i',k)).('h22') = tmp(4);
end

clear data.h_channel_liste
bool = check_if_fibre_prop(h_gnd);
if bool == true
    % means that PMD and CD values were not set to 0
    % and are function of the frequency and then the
    % matrix channel elements (h_{ij}) are different
    % samples to samples.
    for row = 1:2
        for col = 1:2
            for frame = 1:Nframes
                tmp = h_gnd.(sprintf('frame%i',frame)).(sprintf('h%i%i',row,col));
                h_gnd.(sprintf('frame%i',frame)).(sprintf('h%i%i',row,col)) = sum(tmp{1});
            end
        end
    end

else
    % if no PMD nor CD, then the h_{ij} are the same
    % for all samples, so we need only one element
    for row = 1:2
        for col = 1:2
            for frame = 1:Nframes
                tmp = h_gnd.(sprintf('frame%i',frame)).(sprintf('h%i%i',row,col));
                h_gnd.(sprintf('frame%i',frame)).(sprintf('h%i%i',row,col)) = tmp{1}(1);
            end
        end
    end
end
clear tmp

thetas_gnd  = zeros(3,Nframes-FrameChannel);
h11s        = zeros(Nframes-FrameChannel,1);
h12s        = zeros(Nframes-FrameChannel,1);
h21s        = zeros(Nframes-FrameChannel,1);
h22s        = zeros(Nframes-FrameChannel,1);

for k = 1:Nframes-FrameChannel
    h11s(k)         = h_gnd.(sprintf('frame%i',k+FrameChannel)).h11;
    h12s(k)         = h_gnd.(sprintf('frame%i',k+FrameChannel)).h12;
    h21s(k)         = h_gnd.(sprintf('frame%i',k+FrameChannel)).h21;
    h22s(k)         = h_gnd.(sprintf('frame%i',k+FrameChannel)).h22;

    ratio_hij_H     = abs(h12s(k)/h11s(k));
    ratio_hij_V     = abs(h21s(k)/h22s(k));

    thetas_gnd(1,k) = atan(ratio_hij_H)*180/pi;   % [deg]
    thetas_gnd(2,k) = atan(ratio_hij_V)*180/pi;   % [deg]
end

thetas_gnd(3,:)     = mean(thetas_gnd(1:2,:));  % [deg]
FIRgnd.HH           = h11s;
FIRgnd.HV           = h12s;
FIRgnd.VH           = h21s;
FIRgnd.VV           = h22s;

params = {"marker"  ,'square'   ,...
          "size"    , 100        ,...
          "fill"    , ''};

if plot_flag_poincare
    Sgnd   = FIR2Stockes(FIRgnd,params);
end


%%
if plot_flag_poincare
    cd(PathSave{1})
    saveas(f,sprintf("%s --- Poincare.png",filename))
    cd(myInitPath)
end
%%
% ----------------------------------------------
% METRICS
% ----------------------------------------------

Err             = thetas_est-thetas_gnd(3,:);   % [deg]
ErrMean(kdata)  = mean(Err);
ErrStd(kdata)   = std(Err);
ErrRms(kdata)   = ErrStd(kdata)/ErrMean(kdata);

Rs              = get_value_from_filename(myPath,'Rs',Fn{1});
if Rs == 64
    Err         = [zeros(1,20),Err];
elseif Rs == 128
    Err         = [zeros(1,40),Err];
end

params_avg.av_method    = "mirror";
params_avg.av_period    = 5;
params_std.method       = "mirror";
params_std.period       = 5;

Err_mov_avg = moving_average(Err,params_avg);
Err_mov_std = moving_std(Err,params_std);

%%
% ----------------------------------------------
% PLOTS
% ----------------------------------------------
FIRtaps     = linspace(1,FIRlength,FIRlength)-FIRlength/2;

% remove useless dimensions
H_est_11    = squeeze(H_est(1,1,:));
H_est_12    = squeeze(H_est(1,2,:));
H_est_21    = squeeze(H_est(2,1,:));
H_est_22    = squeeze(H_est(2,2,:));

clear H_est
clear('h_est_11_I_j','h_est_12_I_j','h_est_21_I_j','h_est_22_I_j')
clear('h_est_11_Q_j','h_est_12_Q_j','h_est_21_Q_j','h_est_22_Q_j')

H_ests_abs  = abs([H_est_11,H_est_12,H_est_21,H_est_22]);
norm_factor = max(max(H_ests_abs));
H_ests_norm = H_ests_abs/norm_factor;

if plot_flag
    f2 = figure;
        subplot(2,2,1);
            hold on
            plot(FIRtaps,abs(H_ests_norm(:,1)),LineWidth=5,Color='k')
            plot(FIRtaps,abs(H_ests_norm(:,4)),'--',color = ones(1,3)*0.83,LineWidth=2)
            xlabel("filter taps")
            ylabel("amplitude")
            legend("$h_{11}$","$h_{22}$")
            axis([-10,10,0,1])
    
        subplot(2,2,2);
        hold on
            plot(FIRtaps,abs(H_ests_norm(:,2)),LineWidth=5,color = 'k')
            plot(FIRtaps,abs(H_ests_norm(:,3)),'--',color = ones(1,3)*0.83,LineWidth=2)
            xlabel("filter taps")
            legend("$h_{12}$","$h_{21}$")
            axis([-10,10,0,1])

        subplot(2,2,[3,4])
            hold on
            if strcmpi(plot_SOP,'error per frame')
                scatter(frames,thetas_est-thetas_gnd(3,:),100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
                xlabel("frame")
                ylabel("$\hat{\theta}-\theta$ [deg]")

            elseif strcmpi(plot_SOP,'error per theta')
                scatter(thetas_gnd(3,:),thetas_est-thetas_gnd(3,:),100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
                xlabel("$\theta$ [deg]")
                ylabel("$\hat{\theta}-\theta$ [deg]")

            elseif strcmpi(plot_SOP,'comparison per frame')
                plot(frames,thetas_gnd(3,:),'color',[1,1,1]*0.83, LineWidth=5)
                scatter(frames,thetas_est,100,"filled",MarkerEdgeColor="k",MarkerFaceColor='k')
                legend("ground truth","estimation",Location="northwest")
                xlabel("frame")
                ylabel("$\hat{\theta},\theta$ [deg]")
            end

            title(sprintf("Error to ground truth = %.2f +/- %.1f [deg]",ErrMean(kdata),ErrStd(kdata)))
    
    cd(PathSave{1})
    exportgraphics(f2,sprintf("%s.png",filename))
    cd(myInitPath)
    pause(0.25)
    close all

end
cd(PathSave{1})


% M = [Carac,ErrMean,ErrStd,ErrRms];
writematrix(Err.',strcat('Err Theta-',filename,'.csv'))

% HHs     = ifft(FIRest.HH.');
% HVs     = ifft(FIRest.HV.');
% VHs     = ifft(FIRest.VH.');
% VVs     = ifft(FIRest.VV.');
% Firs    = [HHs,HVs,VHs,VVs];
% writematrix(Firs,strcat('Firs-',filename,'.csv'))
cd(myInitPath)


end

cd(PathSave{1})
M           = [Carac,ErrMean,ErrStd,ErrRms];
M(end+1,:)  = [0,median(ErrMean),median(ErrStd),median(ErrRms)];
writematrix(M,strcat('<Err Theta>-',filename,'.csv'))
cd(myInitPath)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   default_plots               
%   import_data                 acceptedFormats                 [allData, allFilenames, allPathnames]
%   get_value_from_filename_in  folderPath,quantity,varargin    out
%   get_number_from_string_in   stringIn,what,varargin          out
%   check_if_fibre_prop         h                               bool
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

function out = get_value_from_filename_in(folderPath,quantity,varargin)

    cd(folderPath)
  
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

function bool = check_if_fibre_prop(h)

    input = h.frame1.h11{1};
    
    if all(input == input(1),'all')
        bool = false;
    else
        bool = true;
    end
end
%-----------------------------------------------------
