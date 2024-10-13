% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 1.1.0
%   Date            : 2024-09-25
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- CHANGELOG -----
%   2024-09-19  (1.0.0) creation
%   2024-09-25  (1.1.0) difference between the cos values
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


thetas  = [-15,-45,-75]*pi/180;
thetas  = [fliplr(thetas),0,-thetas];

SNRsdB  = [3,6,9,12,15,18,21,24,27,30];
SNRsdB  = SNRsdB(1:2:end);
SNRs    = 10.^(SNRsdB/10);

Nths    = length(thetas);
NSNRs   = length(SNRsdB);
Nrea    = 1e5;


dets    = zeros(Nrea,NSNRs,Nths);
Trs     = zeros(Nrea,NSNRs,Nths); % traces
ThHats  = zeros(Nrea,NSNRs,Nths);
ThExt   = zeros(Nrea,NSNRs,Nths);
dthetas = zeros(Nrea,NSNRs,Nths);

elapsed_time = 1e-9;
dnu     = 1e2;
Rs      = 64e9;
var_phi = 2*pi*dnu/Rs;
Nsymbs  = elapsed_time*Rs;


for m = 1:Nrea
    for k = 1:NSNRs
        for j = 1:Nths
            delta   = sqrt(3./SNRs(k));
            dtheta  = -delta+2*delta*rand;
            dthetas(m,k,j) = dtheta;

            phi = sqrt(Nsymbs*var_phi)*randn;

            Theta   = thetas(j)+dtheta;
            R       = [ [cos(Theta),sin(Theta)];...
                        [-sin(Theta),cos(Theta)]]*...
                        exp(1i*phi);

            dets(m,k,j)     = det(R);
            Trs(m,k,j)      = trace(R);
            DR              = R(1,2)-R(2,1);
            ThHats(m,k,j)   = atan(real(DR/Trs(m,k,j)));
            ThExt(m,k,j)    = thetas(j);
        end
    end
end    
    
mean_dets   = real(mean(dets));
std_dets    = real(std(dets));

mean_thHats = mean(ThHats)*180/pi;
std_thHats  = std(ThHats)*180/pi;

mean_TanthHats= mean(tan(ThHats));
std_TanthHats = std(tan(ThHats));


mean_dTanth   = abs(mean(tan(ThHats)-tan(ThExt)));
std_dTanth    = std(tan(ThHats)-tan(ThExt));

mean_dth   = mean(ThHats-ThExt)*180/pi;
std_dth    = std(ThHats-ThExt)*180/pi;



leg_keys        = strings(1,Nths);
colors          = lines(10); % 10 couleurs
line_styles     = {'-', '--', ':', '-.'}; % 4 styles de ligne
markers         = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', 'x'}; % 10 marqueurs différents

for k = 1:Nths
    color_idx       = mod(k-1, size(colors, 1)) + 1;
    line_style_idx  = mod(k-1, length(line_styles)) + 1;
    marker_idx      = mod(k-1, length(markers)) + 1;
    leg_keys(k)     = num2str(round(thetas(k)*180/pi, 0));


%     f1 = figure(1);
%         hold on
%         h2 = plot(SNRsdB, mean_dets(:,:,k),...
%             'Color', colors(color_idx, :), ...
%             'LineStyle', line_styles{line_style_idx}, ...
%             'Marker', markers{marker_idx}, ...
%             'MarkerFaceColor', colors(color_idx, :),...
%             'MarkerEdgeColor', colors(color_idx, :),...
%             'LineWidth', k*0.5,...
%             'DisplayName', leg_keys(k));
%     
%         set(gca, "YScale", 'log')
%         xlabel('SNR [dB]')
%         ylabel('$<\det(R)>$')
%         grid on
%         axis square
%         box on
%         lgd1 = legend();
%         legend boxoff

    f2 = figure(2);
        hold on
        h3 = plot(SNRsdB, std_thHats(:,:,k),...
            'Color', colors(color_idx, :), ...
            'LineStyle', line_styles{line_style_idx}, ...
            'Marker', markers{marker_idx}, ...
            'MarkerFaceColor', colors(color_idx, :),...
            'MarkerEdgeColor', colors(color_idx, :),...
            'LineWidth', k*0.5,...
            'DisplayName', leg_keys(k));
    
%         set(gca, "YScale", 'log')
        xlabel('SNR [dB]')
        ylabel('$\sigma(\hat{\theta})~[deg]$')
        grid on
        axis square
        box on
        lgd2 = legend();
        legend boxoff

    f3 = figure(3);
        hold on
        h3 = errorbar(SNRsdB, mean_dTanth(:,:,k),std_dTanth(:,:,k),...
            'Color', colors(color_idx, :), ...
            'LineStyle', line_styles{line_style_idx}, ...
            'Marker', markers{marker_idx}, ...
            'MarkerFaceColor', colors(color_idx, :),...
            'MarkerEdgeColor', colors(color_idx, :),...
            'LineWidth', k*0.5,...
            'DisplayName', leg_keys(k));
    
        
%         set(gca, "YScale", 'log')
        xlabel('SNR [dB]')
        ylabel('$<\tan(\hat{\theta})-\tan(\theta)>$')
        grid on
        axis square
        box on
        lgd3 = legend();
        legend boxoff
        

    f4 = figure(4);
        hold on
%         TMPpos = mean_dth(:,:,k)>0;
%         TMPneg = mean_dth(:,:,k)<0;
% 
%         TMPk    = mean_dth(:,:,k);

        h4 = errorbar(SNRsdB, mean_dth(:,:,k),std_dth(:,:,k),...
            'Color', colors(color_idx, :), ...
            'LineStyle', line_styles{line_style_idx}, ...
            'Marker', markers{marker_idx}, ...
            'MarkerFaceColor', colors(color_idx, :),...
            'MarkerEdgeColor', colors(color_idx, :),...
            'LineWidth', k*0.5,...
            'DisplayName', leg_keys(k));
%         h4 = plot(SNRsdB, TMPk(TMPpos),...%,std_dth(:,:,k),...
%             'Color', colors(color_idx, :), ...
%             'LineStyle', line_styles{line_style_idx}, ...
%             'Marker', markers{marker_idx}, ...
%             'MarkerFaceColor', colors(color_idx, :),...
%             'MarkerEdgeColor', colors(color_idx, :),...
%             'LineWidth', k*0.5,...
%             'DisplayName', leg_keys(k));
%         h4 = plot(SNRsdB, TMPk(TMPneg),...%,std_dth(:,:,k),...
%             'Color', colors(color_idx, :), ...
%             'LineStyle', line_styles{line_style_idx}, ...
%             'Marker', markers{marker_idx}, ...
%             'MarkerFaceColor', colors(color_idx, :),...
%             'MarkerEdgeColor', colors(color_idx, :),...
%             'LineWidth', k*0.5,...
%             'DisplayName', leg_keys(k));
    
        
    %     set(gca, "YScale", 'log')
        xlabel('SNR [dB]')
        ylabel('$<\hat{\theta}-\theta>$')
        grid on
        axis square
        box on
        lgd4 = legend();
        legend boxoff

end

f4 = figure(4);
hold on
plot(SNRsdB,5*ones(size(SNRsdB)),'k', LineWidth=3,DisplayName='$+5~[deg]$')
plot(SNRsdB,-5*ones(size(SNRsdB)),'k', LineWidth=3,DisplayName='$-5~[deg]$')


% set(f1, 'Position', [0.0198, 0.0009, 0.5255, 0.8824])
set(f2, 'Position', [0.0198, 0.0009, 0.5255, 0.8824])
set(f3, 'Position', [0.0198, 0.0009, 0.5255, 0.8824])

% title(lgd1,'$\theta~[deg]$');
title(lgd2,'$\theta~[deg]$');
title(lgd3,'$\theta~[deg]$');