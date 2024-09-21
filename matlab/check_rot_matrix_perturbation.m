% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.0.0
%   Date            : 2024-09-19
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGELOG -----
%   2024-09-19  (1.0.0) creation
% 
% ----- INPUTS -----
% ----- EVOLUTIONS -----%
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


clear
close all
clc


thetas  = linspace(0,pi,3);
SNRsdB  = linspace(3,30,5);
SNRs    = 10.^(SNRsdB/10);

Nths    = length(thetas);
NSNRs   = length(SNRsdB);
Nrea    = 10;


dets    = zeros(Nrea,NSNRs,Nths);
Trs     = zeros(Nrea,NSNRs,Nths);
ThHats  = zeros(Nrea,NSNRs,Nths);
xs      = zeros(Nrea,NSNRs,Nths);

for m = 1:Nrea
    for k = 1:NSNRs
        for j = 1:Nths
            N0      = 1./SNRs(k);
            noise   = sqrt(N0)*randn(2);
            R       = [ [cos(thetas(j)),sin(thetas(j))];...
                        [-sin(thetas(j)),cos(thetas(j))]];
            Rn      = R+noise;

            dets(m,k,j)     = det(Rn);
            Trs(m,k,j)      = trace(Rn);
            ThHats(m,k,j)   = real(acos(Trs(m,k,j)/2/sqrt(dets(m,k,j))));
            xs(m,k,j)       = trace(R^(-1)*noise)+det(R^(-1)*noise);
        end
    end
end    
    
mean_dets   = mean(dets);
std_dets    = std(dets);
mean_thHats = mean(ThHats);
std_thHats  = std(ThHats);
mean_xs     = mean(xs);

Mdets_res   = zeros(Nths,NSNRs);
Sdets_res   = zeros(Nths,NSNRs);
MThHat_res  = zeros(Nths,NSNRs);
SThHat_res  = zeros(Nths,NSNRs);
Mxs_res     = zeros(Nths,NSNRs);

for k = 1:Nths
    Mdets_res(k,:)  = mean_dets(:,:,k);
    Sdets_res(k,:)  = std_dets(:,:,k);
    MThHat_res(k,:) = mean_thHats(:,:,k);
    SThHat_res(k,:) = std_thHats(:,:,k);
    Mxs_res(k,:)    = std_thHats(:,:,k);
end

% ref_Ths         = repmat(thetas,[1,length(SNRs)]);
% err_ThHats      = 

% figure
% for k = 1:Nths
%     subplot(1,3,1)
%         hold on
%         plot(SNRsdB,abs(1-mean_dets(:,:,k)))
%         set(gca,"YScale",'log')
%     subplot(1,3,2)
%         hold on
%         errorbar(SNRsdB,mean_thHats(:,:,k),std_thHats(:,:,k))
%     subplot(1,3,3)
%         hold on
%         plot(SNRsdB,std_thHats(:,:,k))
%         set(gca,"YScale",'log')
% end