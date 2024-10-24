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


Nrea    = 100;
Np      = 10000;
f0      = 2e1;
fs      = 100*f0;
dt      = 1/fs;
t       = linspace(0,(Np-1)*dt,Np);
df      = 1/t(end);
x       = 1/sqrt(2)*sin(2*pi*f0*t);%+cos(2*pi*f0/2*t)+cos(2*pi*f0/3*t).*sin(2*pi*f0/4*t);
Px      = mean(abs(x).^2);
SNRs    = [2,5,10,20,50,100];

for j = 1:length(SNRs)
    SNR     = SNRs(j);
    Rxx     = xcorr(x,x);
    
    y       = zeros(Nrea,Np);
    Ryy     = zeros(Nrea,2*Np-1);
    Rxn     = zeros(Nrea,2*Np-1);
    Rnx     = zeros(Nrea,2*Np-1);
    Rnn     = zeros(Nrea,2*Np-1);
    Rxnoise = zeros(Nrea,2*Np-1);
    
    for k = 1:Nrea
        n           = sqrt(1/SNR)*randn(1,Np);
        y(k,:)      = x+n;
    
        Ryy(k,:)    = xcorr(y(k,:),y(k,:));
        Rxn(k,:)    = xcorr(x,n);
        Rnx(k,:)    = xcorr(n,x);
        Rnn(k,:)    = xcorr(n,n);
    end
    
    PRxx    = mean(abs(Rxx).^2);
    RYY     = mean(Ryy);
    RXN     = mean(Rxn);
    RNX     = mean(Rnx);
    RNN     = mean(Rnn);
    
    Rxnoise = RNX+RXN;
    % figure
    % hold on
    % plot(t,x)
    % plot(t,y)
    
%     figure
%         subplot(2,2,1)
%             plot(Rxx)
%         subplot(2,2,3)
%             tmp =RNN(RNN~=max(RNN));
%             semilogy(abs(tmp))
%             title(sprintf('SNR = %d,%.2e, max = %.1e',...
%                 SNR,mean(abs(tmp).^2),max(RNN)))
%         subplot(2,2,2)
%             plot(RXN)
%         subplot(2,2,4)
%             plot(RNX)

    tmp =RNN(RNN~=max(RNN));
    fprintf('\nSNR = %d\t %.2e\t max = %.1e',...
                SNR,mean(abs(tmp).^2),max(RNN))
end
%         hold on
%         plot(Rxn/sqrt(PRxx))
%         plot(Rnx/sqrt(PRxx))


SXN     = fft(RXN)/sqrt(2*Np); % factor to check Parseval Plancherel
PSXN    = sum(abs(SXN).^2*df);


%%
SNRs        = [2,5,10,20,50,100];
Maxs        = [5000,2000,1000,500,200,100]/2;
Floors      = [12.5,2,0.5,0.125,0.02,0.005]/2;

SNRsdB      = 10*log10(SNRs);
MaxsdB      = 10*log10(Maxs);
FloorsdB    = 10*log10(Floors);

Nps         = [500,1000,5000,10000,50000];
FNp         = [2.5,5,25,50,250];