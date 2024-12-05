% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.2.1
%   Date            : 2024-11-26
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
%   2024-11-05  (2.1.1) adding AWGN
%                       IMPORT_DATA (1.1.2) raise error if no files
%   2024-11-06  (2.2.0) [REMOVED] IMPORT_DATA
%   2024-11-26  (2.2.1) managing no files
%
% ----- MAIN IDEA -----
% Used to plot according to the batch size the BER and success rate
%           OR
% the AWGN curve
% data are moved in MATLAB/error_estimation folder
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

%% import data
rst
caps.log.Date   = '24-11-27';
entropy         = 6;


caracs0     = {'mimo','vae'}; % do not remove
caracs1   = {'Rs',64}; %d
caracs2   = {'SNR_dB',22}; %d
% caracs2   = {'SNR_dB',[22]}; %d 
% caracs3   = {'NSbB',[250]}; %d
% caracs3   = {'dnu',[1]}; % {%d_, [X]; .1f [X]/10}
% caracs3   = {'CFO',[1,5,10]}; %d
% caracs4   = {'NSbF',20}; %.1f
% caracs4   = {'PhEnd',10}; %d
% caracs4   = {'CFO',10}; %.1f
% caracs5   = {'PhEnd',90}; %d
% caracs3   = {'NSbB',[50,100,150,200,250,300,350,400,450]}; %d
caracs3   = {'vsop',[50,100,500,1000]}; %.1f
caracs4   = {'PhEnd',0}; %d
caracs5   = {'lr',0.75}; %.2f



% SNR         = [25,24,23,22,21,20,19,18,17,16,15];

AWGN        = 0;
theta_or_phi_Ftrain = "theta";
Ftrain      = 30; % {CFO: 1-cma, 30-vae}
fec_limit   = 2.8e-2;
time_est    = 0;
plot_T      = 0;
% 7 = 3 caracs + ser + time conv + time frame + frame conv

Ntasks  = length(caracs1{2})*length(caracs2{2})*length(caracs3{2})*length(caracs4{2});
count   = 0;
countT  = 0;

res = struct();
fn_keep = strings(1,length(caracs1{2})*length(caracs2{2})*length(caracs3{2}));


patch = 51;

for ncarac1 = 1:length(caracs1{2})
    fprintf("\n\t %s = %.1f\n",caracs1{1},caracs1{2}(ncarac1))
    for ncarac2 = 1:length(caracs2{2})
        fprintf("\t\t %s = %.1f\n",caracs2{1},caracs2{2}(ncarac2))
        for ncarac3 = 1:length(caracs3{2})
            fprintf("\t\t %s = %.1f\n",caracs3{1},caracs3{2}(ncarac3))
            for ncarac4 = 1:length(caracs4{2})
                for ncarac5 = 1:length(caracs5{2})
                    cd(strcat('../python/data-',caps.log.Date,"/csv"))
            
                    caracs         = [sprintf("%s %d",caracs1{1},caracs1{2}(ncarac1));...
                                      sprintf("%s %d",caracs2{1},caracs2{2}(ncarac2));...
                                      sprintf("%s %d ",caracs3{1},caracs3{2}(ncarac3));...
                                      sprintf("%s %d",caracs4{1},caracs4{2}(ncarac4));...
                                      sprintf("%s %.2f",caracs5{1},caracs5{2}(ncarac5));...
                                      sprintf("%s %s",caracs0{1},caracs0{2});...
                                                ];

                    [allData,caps]          = import_data({'.csv'},caps,caracs);
                    caps.log.myInitPath     = pwd();
%                     fprintf('\t Nfiles = %d\n',caps.log.Nfiles)
                    
                    if caps.log.Nfiles ~= 0
                        matrix_tmp          = zeros(caps.log.Nfiles,8); % 8 if carac3, 7 otherwise
                        location            = zeros(1,caps.log.Nfiles);
                        Niter               = patch;%allData{1}.iteration(end);
                        bers                = zeros(Niter,caps.log.Nfiles);
                        if ~AWGN
                            thetas          = zeros(Niter,caps.log.Nfiles);
                        end
                        if time_est
                            dt              = zeros(Niter,caps.log.Nfiles);
                        end
                        if sum(contains(allData{1}.Properties.VariableNames,'Phis'))
                            phis            = zeros(Niter,caps.log.Nfiles);
                        end
    
                        for k = 1:caps.log.Nfiles
    
                            if ~AWGN
                                if strcmpi(theta_or_phi_Ftrain,'theta')
                                    tmpTP    = allData{k}.Thetas(1:patch);
                                else
                                    tmpTP    = allData{k}.Phis(1:patch);
                                end
                                Nf      = length(tmpTP);
                                Ftrain  = Nf-sum(tmpTP ~= 0);
    
                                if Ftrain == Nf
                                    tmpP    = allData{k}.Phis;
                                    Ftrain  = Nf-sum(tmpP ~= 0);
                                    NFrChnl = Nf-Ftrain;
                                else
                                    NFrChnl = Nf-Ftrain;
                                end
    
                                Fr_avg  = ceil(NFrChnl/2);
                            else
                                Fr_avg  = Ftrain+1;
                            end

                            if ~exist('Ftrain') && ~AWGN
                                thetas(:,k) = allData{k}.Thetas(1:patch);
                            end

                            bers(:,k)       = allData{k}.SER(1:patch)/entropy;

                            
                            if time_est
                                dt(:,k)     = allData{k}.dt(1:patch);
                            end
                
                            if sum(contains(allData{k}.Properties.VariableNames,'Phis'))
                                phis(:,k)   = allData{1}.Phis(1:patch);
                            end
                
                            % estimation of convergence rate
%                             bers_2          = bers(FrChnl+1:end,k);
%                             bers_2          = bers(end-FrChnl:end,k);
                            bers_2          = bers(end-Fr_avg:end,k);
                            bers_3          = bers_2(bers_2<=fec_limit);
                            npass           = length(bers_3);
                            rate_conv       = npass/length(bers_2);
    
                            x1              = caracs1{2}(ncarac1);
                            x2              = caracs2{2}(ncarac2);
                            x12             = caracs3{2}(ncarac3);
                            x3              = caracs4{2}(ncarac4);
                            x4              = mean(bers_2);
                            x4b             = std(bers_2);
                            x5              = rate_conv;
    
                            if time_est
                                x6          = mean(mean(dt));
                            else
                                x6          = nan;
                            end
    
                            matrix_tmp(k,:) = [x1,x2,x12,x3,x4,x4b,x5,x6];
                        end
                        
                        if size(matrix_tmp,1) ~= 1
                            matrix = mean(matrix_tmp);
                        else
                            matrix = matrix_tmp;
                        end
                        cd(caps.log.myRootPath)
                        
                        if ~exist('T','var')
                            T = array2table(matrix,'VariableNames', ...
                                {caracs1{1},caracs2{1},caracs3{1},caracs4{1},'meanBER','stdBER','RateConv','TIMEframe'});
    
                        else
                            Ttmp = array2table(matrix,'VariableNames', ...
                                {caracs1{1},caracs2{1},caracs3{1},caracs4{1},'meanBER','stdBER','RateConv','TIMEframe'});
    
                            T = [T;Ttmp];
                        end
                        
                    else
                        cd(caps.log.myRootPath)
                        continue
                    end
                    count = count + 1;
                    fprintf('Progress: %.1f/100 --- %s - %s - %s - %s - %s - %s\n',...
                        round(count/Ntasks*100,1),...
                        caracs');
                end
    
                if caps.log.Nfiles ~= 0
                    countT = countT +1;
                    res.(sprintf('T%d',countT)) = T;
                    filename = char(caps.log.Fn{1});
%                     writetable(T,filename)
                else
                    continue
                end
                
                fn_keep(1,countT) = filename;
            end
        end
    end
end

if ~isempty('T') && countT ~= 0
    cd error_estimation/
    writetable(T,filename)
end

if plot_T
f = figure;
hold all
grid on
line_styles = {'-', '--', ':', '-.'};
markers     = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', 'x'};


set(gca, 'LineStyleOrder', {'-', '--', ':', '-.'}, 'NextPlot', 'add');
if ~AWGN
    xlabel('$\mathbf{N_{Symb,Batch}}$')
    xlim([25,525])
else
    xlabel('$\mathbf{SNR~[dB]}$')
    xlim([14.5,25.5])
end
if ~AWGN
    ttRC    = strings(length(fieldnames(res)),3);
    ttBER   = strings(length(fieldnames(res)),3);
    
    ttRC(:,1)    = "CR - ";
    ttBER(:,1)   = "BER - ";
end

if ~AWGN
    for j = 1:length(fieldnames(res))

        line_style_idx  = mod(j-1, length(line_styles)) + 1;
        marker_idx      = mod(j-1, length(markers)) + 1;
    
        T           = res.(sprintf('T%d',j));
        what        = {char("SNR_dB")};%{'vsop','CFO'};
        index       = zeros(1,length(what));
        what_val    = zeros(1,length(what));
        filename    = fn_keep{j};

        for k = 1:length(what)
    
            index(k)       = findstr(filename,what{k});
            
            if strcmpi(what{k},'lr')
                what_val(k) = str2double(filename(index(k)+length(what{k}):index(k)+6));
                what_val(k) = what_val(k)/1e3;
            else
                is          = index(k)+length(what{k});
                ie          = index(k)+length(what{k})+3;
                what_val(k) = str2double(filename(is:ie));
            end
        end

        for k = 1:length(what)
            ttRC(j,k+1)   = sprintf('%s - %.1f',what{k},what_val(k));
            ttBER(j,k+1)  = sprintf('%s - %.1f',what{k},what_val(k));
        end
    end

    yyaxis left
    ylim([1e-1,1.01])
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
    set(gca,'Yscale','log')
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
            ylim([5e-3,5e-1])
        else
            ylim([1e-5,1])
        end
else
    errorbar(T.SNR_dB,T.meanBER,T.stdBER,...
    'color', 'k',...
    'MarkerFaceColor','k',...
    'MarkerEdgeColor','k',...
    'MarkerSize',15,...
    'LineWidth', k)
    ylim([1e-5,1e-1]*5)
end
set(gca,'Yscale','log')

if ~AWGN
    ax = gca;
    ax.YAxis(1).Color = 'k';
    ax.YAxis(2).Color = 'b';
    
    yyaxis left
        h = plot([min(caracs4{2}),max(caracs4{2})],[1,1]*fec_limit, ...
            '-r',LineWidth=5);
end
h = plot([15,25],[1,1]*fec_limit, ...
    '-r',LineWidth=5);
annotation('textbox',...
[0.56,0.61,0.35,0.04],'Color',[1 0 0],...
'String',{'$\mathbf{FEC~limit = 2.8\cdot 10^{-2}}$'},...
'Interpreter','latex',...
'FontWeight','bold',...
'FontSize',20,...
'FitBoxToText','off',...
'EdgeColor','none');
set(h,"DisplayName",'')

if ~AWGN
    lgd         = legend(Location="southoutside",NumColumns=2);
    entries     = get(lgd, 'String');
    newEntries  = entries(1:end-1);
    set(lgd, 'String', newEntries);
end

if AWGN
    lgd = legend(sprintf("%s  - Rs %d [GBd]",caracs0{2},caracs1{2}));
end
box on
legend boxoff
set(gcf, 'Position', [0.0198,0.0009,0.5255,0.8824])
set(lgd,'Interpreter','latex')

if ~AWGN
    if SNR == 17 && caracs3{2} == 64
        title(lgd,'\textbf{@SNR = 17 [dB], $R_s =$ 64 [GBd], CFO $\times$ 10 [kHz], $v_{SoP} \times$ 10 [krad/s]}');
    
    elseif SNR == 17 && caracs3{2} == 128
        title(lgd,'\textbf{@SNR = 17 [dB], $R_s =$ 128 [GBd], CFO $\times$ 10 [kHz], $v_{SoP} \times$ 10 [krad/s]}');
    
    elseif SNR == 25 && caracs3{2} == 64
        title(lgd,'\textbf{@SNR = 25 [dB], $R_s =$ 64 [GBd], CFO $\times$ 10 [kHz], $v_{SoP} \times$ 10 [krad/s]}');
    
    elseif SNR == 25 && caracs3{2} == 128
        title(lgd,'\textbf{@SNR = 25 [dB], $R_s =$ 128 [GBd], CFO $\times$ 10 [kHz], $v_{SoP} \times$ 10 [krad/s]}');

    end
end



saveas(f,[filename(1:end-3),'fig'])
saveas(f,[filename(1:end-3),'svg'])
saveas(f,[filename(1:end-3),'png'])
end