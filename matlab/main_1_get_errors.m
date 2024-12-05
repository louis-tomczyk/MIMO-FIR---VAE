% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.1.0
%   Date            : 2024-11-11
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
%   2024-10-26  (2.0.1) caracs5
%   2024-10-28  (2.0.2) automatic folder selection
%   2024-11-08  (2.0.3) logistics
%   2024-11-11  (2.1.0) std correction available
% 
% ----- MAIN IDEA -----
%   get estimation errors of the angles theta and phi
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


%% MAINTENANCE
rst
caps.log.Date           = '24-11-27';

caracs0     = '<ErrTh>';            % {<ErrTh>,<ErrPh>}
caracs6     = {'','fft'};              % {'','mat','fft'}


caracs1   = {'Rs',64}; %d
caracs2   = {'SNR_dB',18}; %d
% caracs2   = {'SNR_dB',[22]}; %d 
% caracs3   = {'NSbB',[250]}; %d
% caracs3   = {'dnu',[0.5,1,2,5,10]}; % {%d_, [X]; .1f [X]/10}
% caracs3   = {'CFO',[1,5,10]}; %d
% caracs4   = {'NSbF',20}; %.1f
% caracs4   = {'PhEnd',10}; %d
% caracs4   = {'CFO',10}; %.1f
% caracs5   = {'PhEnd',90}; %d
% caracs3   = {'NSbB',[50,100,150,200,250,300,350,400,450]}; %d
caracs3   = {'vsop',[50,100,500,1000]}; %d
caracs4   = {'PhEnd',0}; %d
caracs5   = {'lr',1}; %.2f

% copy the names of the caracsX
table_varnames = {'Rs','SNR_dB','vsop','PhEnd','lr','ndata','median','std','rms'};
% table_varnames = {'Rs','SNR_dB','dnu','NSbF','lr','ndata','median','std','rms'};
% table_varnames = {'Rs','SNR_dB','dnu','NSbF','lr','median','std','rms'};

caps.save.errs          = 1;
caps.save.mean_errs     = 1;
caps.what_carac         = caracs1{1};
caps.correct_std.do     = 0;

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

                    if contains(caracs0,'Ph')
                        cd(strcat('../python/data-',caps.log.Date,"/err/sum_up_phi"))
                    else
                       cd(strcat('../python/data-',caps.log.Date,"/err/sum_up_theta"))
                    end

                    caps.log.myInitPath     = pwd();
    
                    caracs         = [sprintf("%s %d",caracs1{1},caracs1{2}(ncarac1));...
                                      sprintf("%s %d",caracs2{1},caracs2{2}(ncarac2));...
                                      sprintf("%s %d ",caracs3{1},caracs3{2}(ncarac3));...
                                      sprintf("%s %d",caracs4{1},caracs4{2}(ncarac4));...
                                      sprintf("%s %.2f",caracs5{1},caracs5{2}(ncarac5));...
                                      sprintf("%s%s",caracs6{1},caracs6{2});...
                                      ];
            
                    [allData,caps]  = import_data({'.csv'},caps,caracs); % {,manual selection}
                    

                    if ~isempty(allData)
                        cd(caps.log.myInitPath)
                        Errs(row,:) = allData{1}(end,:);
                        % the end line is the median value of the errors of the previous lines
                        fname = caps.log.Fn{1};
                        row = row+1;
                    end
                    cd(caps.log.myRootPath)

                end % carac5
            end % carac4
        end % carac3
    end % carac2
end % carac1

if exist("Errs")
    Errs.Properties.VariableNames = table_varnames;
    if caps.correct_std.do
        caps.correct_std.params.nref = min(Errs.ndata);
        Errs = correct_std(Errs,caps);
    end 
end

if exist("Errs")
    Errs.Properties.VariableNames = table_varnames;
    if ~exist("error_estimation/",'dir')
        mkdir(tmp)
    end
    cd error_estimation/
    
    writetable(Errs,fname)
else
    fprintf("\n\t no files found at all ---> no output file\n")
end
cd(caps.log.myRootPath)




function y = correct_std(Errs,caps)

    nref    = caps.correct_std.params.nref;
    diff_n  = abs(Errs.ndata-nref);
    corr    = ones(size(diff_n));

    for k = 1:length(diff_n)
        if diff_n(k) ~= 0
            corr(k) = sqrt(diff_n(k));
        else
            corr(k) = 1;
        end
    end

    y   = corr.*Errs.std;
end
