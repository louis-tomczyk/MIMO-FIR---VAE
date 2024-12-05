% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.5.0
%   Date            : 2024-11-11
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
%   2024-11-05  (2.4.0) [REMOVED] IMPORT_DATA moved to lib
%   2024-11-07  (2.4.1) caracs format specification detailed in
%                           'parameters_explanations' (1.1.0)
%   2024-11-08  (2.4.2) managing existence of sum_up_* folders
%   2024-11-11  (2.5.0) adding the number of data in the saved matrices
%   2024-11-26  (2.6.0) warning no channel along with EXTRACT_INFOS (1.2.0)
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
caps.log.Date           = '24-11-27';
caps.plot.fir           = 0;
caps.plot.poincare      = 0;
caps.plot.SOP.xlabel    = 'comparison per frame';   % {'error per frame','error per theta''comparison per frame'}
caps.plot.phis.xlabel   = 'comparison per batch';
caps.method.thetas      = 'mat';                    % {fft, mat, svd}
caps.method.phis        = 'eig';
caps.save.errs          = 1;
caps.save.mean_errs     = 1;
caps.save.data          = 0;

nrea                    = 10;   % used only for progression bar
count                   = 0;



caracs1   = {'Rs',64}; %d
caracs2   = {'SNR_dB',[18,19,22,23]}; %d
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
caracs5   = {'lr',1}; %.2f


caps.what_carac         = caracs1{1};

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
    
                    caracs         = [sprintf("%s %d",caracs1{1},caracs1{2}(ncarac1));...
                                      sprintf("%s %d",caracs2{1},caracs2{2}(ncarac2));...
                                      sprintf("%s %d ",caracs3{1},caracs3{2}(ncarac3));...
                                      sprintf("%s %d",caracs4{1},caracs4{2}(ncarac4));...
                                      sprintf("%s %.2f",caracs5{1},caracs5{2}(ncarac5))];
            
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
                        if ~isfield(caps.log,"warning_no_channel")

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
                            % 4 = ndata, mean, std, rms
                                Mthetas         =  zeros(caps.log.Nfiles+1,length(caracs)+4);
                                if caps.phis_est
                                    Mphis       =  zeros(caps.log.Nfiles+1,length(caracs)+4);
                                end
                            end
        
                            Mthetas(kdata,:)    = [caracs1{2}(ncarac1),...
                                                   caracs2{2}(ncarac2),...
                                                   caracs3{2}(ncarac3),...
                                                   caracs4{2}(ncarac4),...
                                                   caracs5{2}(ncarac5),...
                                                   caps.NFrames.Channel,...
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
                                                   caps.Batches.array(end),...
                                                    metrics.phis.ErrMedian(kdata),...
                                                    metrics.phis.ErrStd(kdata),...
                                                    metrics.phis.ErrRms(kdata)];
                
                            end
                        
                
                            count       = count + 1;
                            fprintf('Progress: %.1f/100 --- %s - %s - %s- %s- %s\n',...
                                round(count/NfilesTot*100,1),...
                                caracs');
                        else
                            continue
                        end
                    end
                    
                    
                    if caps.save.mean_errs && caps.log.Nfiles > 0 && ~isfield(caps.log,"warning_no_channel")
                        cd ../thetas
                        Mthetas(end,:)  = [caracs1{2}(ncarac1),...
                                           caracs2{2}(ncarac2),...
                                           caracs3{2}(ncarac3),...
                                           caracs4{2}(ncarac4),...
                                           caracs5{2}(ncarac5),...
                                          caps.NFrames.Channel,...
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
                            'ndata','mean','std','rms'});

                        cd ../
                        if ~isfolder("sum_up_theta")
                            mkdir sum_up_theta
                        end
                        cd sum_up_theta
                        writematrix(real(Mthetas),str_tmp)
                        cd ../thetas

                        if caps.phis_est
                            cd ..

                            Mphis(end,:)= [caracs1{2}(ncarac1),...
                                           caracs2{2}(ncarac2),...
                                           caracs3{2}(ncarac3),...
                                           caracs4{2}(ncarac4),...
                                           caracs5{2}(ncarac5),...
                                            caps.Batches.array(end),...
                                               median(Mphis(1:end-1,end-2)),...
                                               median(Mphis(1:end-1,end-1)),...
                                               median(Mphis(1:end-1,end))];
                            str_tmp         = strcat( ...
                                                sprintf('<ErrPh>-%s-',caps.method.phis), ...
                                            caps.log.filename(11:end),'.csv'); % 11 == 'matlabX - '
                            if ~isfolder("sum_up_phi")
                                mkdir sum_up_phi
                            end
                            cd sum_up_phi
                                writematrix(Mphis,str_tmp)
                            cd ..
                        end
                    end

                    cd(caps.log.myRootPath)
                end
            end
        end
    end
end