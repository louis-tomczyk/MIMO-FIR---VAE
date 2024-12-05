% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.0
%   Date            : 2024-09-21
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-09-21  (1.0.0)
%   2024-11-06  (1.1.0) - [REMOVED] IMPORT_DATA, moved to lib
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

rst

caps.log.Date   = '24-11-07';
caps.log.warning_nfiles = 0;

caracs1         = {"SNR_dB",[17,18,21,22]};
% caracs2         = {"CFO",[1,2,5,10,20,50]};
caracs2         = {'Rs',[64,128]};
% caracs3         = {"vsop",[10,20,50,100]};
caracs3         = {"dnu",[5,10,20,50]};
caracs4         = {'NSbF',[10,20]};
caracs5         = {'ThEnd',0};
caracs6         = {"NSbB",250};
what            = "csv";   % {mat,csv,log}

if strcmpi(what,'mat')
    fprintf(join(["\t,","\t,","\t,","\t,","\t,","\t,","\t,","\t,","\t,","\t,","\t,","\t,","\t,","mat\n"]))
else
    fprintf("%s\n",what)
end

cnt = 0;
for ncarac1 = 1:length(caracs1{2})
    for ncarac2 = 1:length(caracs2{2})
        for ncarac3 = 1:length(caracs3{2})
            for ncarac4 = 1:length(caracs4{2})
                for ncarac5 = 1:length(caracs5{2})
                    for ncarac6 = 1:length(caracs6{2})
                        cd(strcat('../python/data-',caps.log.Date,sprintf("/%s/",what),''))
                
                        cnt = cnt+1;
                        % add space if [X,X0] like dnu 5 & dnu 50
                        caracs         = [sprintf("%s %d",caracs1{1},caracs1{2}(ncarac1));...
                                          sprintf("%s %d",caracs2{1},caracs2{2}(ncarac2));...
                                          sprintf("%s %d ",caracs3{1},caracs3{2}(ncarac3));...
                                          sprintf("%s %.1f",caracs4{1},caracs4{2}(ncarac4));...
                                          sprintf("%s %d",caracs5{1},caracs5{2}(ncarac5));...
                                          sprintf("%s %d",caracs6{1},caracs6{2}(ncarac6));...
                                                    ];
                       
                        [allData,caps]          = import_data({sprintf('.%s',what)},caps,caracs);
                        caps.log.myInitPath     = pwd();
        
                        cd(caps.log.myRootPath)
                        if strcmpi(what,'mat')
                            fprintf("%s",join(caracs))
        
                            fprintf(",\tNfiles %i\n",caps.log.Nfiles)
                        else
                            fprintf("%i\n",caps.log.Nfiles)
                        end
                    end % carac6
                end % carac5
            end % carac4
        end % carac3
    end % carac2
end % carac1