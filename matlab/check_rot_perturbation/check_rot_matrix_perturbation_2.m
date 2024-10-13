% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.0.0
%   Date            : 2024-09-30
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGELOG -----
%   2024-09-30  (1.0.0) creation
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

rst
% 
% 
% Nt  = 8;
% t   = linspace(0, pi, Nt);
% SNR = linspace(Nt, 1000, 6);
% 
% dt  = zeros(Nt, 6);
% err = zeros(Nt, 6);
% 
% for k = 1:length(t)
%     for j = 1:length(SNR)
%         
%         dt(k, j)  = sin(t(k)) / sqrt(SNR(j));
% %         err(k, j) = abs((cos(t(k) + dt(k, j)) - cos(t(k))) / cos(t(k))) * 100;
%         err(k, j) = abs((cos(t(k) + dt(k, j)) - cos(t(k))));
%         
%     end
% end
% 
% figure('Position', [0.0198, 0.0009, 0.5255, 0.8824])
% hold on;
% for j = 1:length(t)
%     plot(SNR, err(j, :), 'DisplayName', sprintf('%d', round(t(j) * 180 / pi)), 'LineWidth', exp(-j / 10));
% end
% 
% legend('Location', 'southwest', 'NumColumns', 1,'Position', [0.85 0.2 0.1 0.6]); % Légende à gauche
% xlabel('SNR');
% ylabel('Error [percent]');
% set(gca,'Xscale','log')
% set(gca,'Yscale','log')



Nt  = 100;
t   = linspace(0,pi, Nt);
SNR = linspace(Nt, 1000, 6);

dt  = zeros(Nt, 6);
err = zeros(Nt, 6);


for k = 1:length(t)
    for j = 1:length(SNR)
        
        dt(k, j)  = sin(t(k)) / sqrt(SNR(j));
%         err(k, j) = (cos(t(k) + dt(k, j)) - cos(t(k))) / cos(t(k)) * 100;
        err(k, j) = acos(cos(t(k) + dt(k, j))) - acos(cos(t(k)));
        
    end
end


figure('Position', [0.0198, 0.0009, 0.5255, 0.8824])
for k =1:length(SNR)
    hold on
    plot(t*180/pi, err(:,k),DisplayName=sprintf('SNR = %d',SNR(k)));
end
xlabel('$\theta$ [deg]');
ylabel('$\cos(\theta+d\theta)-\cos(\theta)$');
legend()
