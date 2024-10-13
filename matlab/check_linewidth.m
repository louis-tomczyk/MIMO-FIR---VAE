% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.0.0
%   Date            : 2024-10-07
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGE LOG -----
%   2024-10-07  (1.0.0)
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

Nrea    = 50;
Nt      = 5e6;
dnu     = 1e5;
fs      = 1e9;
dt      = 1/fs;
f0      = 1e6;
t       = 0:dt:(Nt-1)*dt;
df      = 1/(t(end));
f       = -(Nt/2-1)*df:df/2:(Nt/2)*df;

STD     = 2*pi*dnu/fs;

phi     = zeros(Nrea,Nt);
x      = zeros(Nrea,Nt);

Rxx     = zeros(Nrea,2*Nt-1);
Sxx      = zeros(Nrea,2*Nt-1);

for k = 1:Nrea
    disp(k)
    phi(k,:)     = cumsum(STD^2*randn(1,Nt));
    x(k,:)       = cos(2*pi*f0*t+phi(k,:));
    
    Rxx(k,:)     = xcorr(x(k,:),x(k,:),"normalized");
    Sxx(k,:)     = fftshift(fft(Rxx(k,:))/Nt);
end

SxxMean = mean(Sxx);


alpha = 500;
fmid = f0/df*2;
fmin = fmid-fmid/alpha;
fmax = fmid+fmid/alpha;
% plot(f,10*log10(abs(SxxMean).^2))
plot(f/1e3,abs(SxxMean).^2)


xlim([fmin,fmax]*df/2/1e3)
% set(gca,'Xscale','log')



