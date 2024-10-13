figure
t = tiledlayout(2,2,'TileSpacing','Compact');
hold on
legendEntries   = [];
colors          = lines(10); % 10 couleurs
line_styles     = {'-', '--', ':', '-.'}; % 4 styles de ligne
markers         = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', 'x'}; % 10 marqueurs différents

for k = 1:Nths
    color_idx       = mod(k-1, size(colors, 1)) + 1;
    line_style_idx  = mod(k-1, length(line_styles)) + 1;
    marker_idx      = mod(k-1, length(markers)) + 1;
    leg_keys(k)     = num2str(round(thetas(k)*180/pi, 0));
    
    % Subplot 1
    nexttile(1)
    hold on
    h1 = plot(SNRsdB, abs(1-mean_dets(:,:,k)), ...
        'Color', colors(color_idx, :), ...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx}, ...
        'MarkerFaceColor', colors(color_idx, :),...
        'MarkerEdgeColor', colors(color_idx, :),...
        'LineWidth', k*0.5,...
        'DisplayName', leg_keys(k));

    set(gca, "YScale", 'log')
    xlabel('SNR [dB]')
    ylabel('$|1-det|$')
    grid on
    axis square
    box on

    % Subplot 2
    nexttile(2)
    hold on
    h2 = errorbar(SNRsdB, mean_CthHats(:,:,k), std_CthHats(:,:,k),...
        'Color', colors(color_idx, :), ...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx}, ...
        'MarkerFaceColor', colors(color_idx, :),...
        'MarkerEdgeColor', colors(color_idx, :),...
        'LineWidth', k*0.5,...
        'DisplayName', leg_keys(k));

    xlabel('SNR [dB]')
    ylabel('$<\hat{\theta}>~[deg]$')
    grid on
    axis square
    box on

    % Subplot 3
    nexttile(3)
    hold on
    h3 = plot(SNRsdB, std_thHats(:,:,k)*180/pi,...
        'Color', colors(color_idx, :), ...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx}, ...
        'MarkerFaceColor', colors(color_idx, :),...
        'MarkerEdgeColor', colors(color_idx, :),...
        'LineWidth', k*0.5,...
        'DisplayName', leg_keys(k));

    set(gca, "YScale", 'log')
    xlabel('SNR [dB]')
    ylabel('$\sigma(\hat{\theta})~[deg]$')
    grid on
    axis square
    box on

    % Subplot 4
    nexttile(4)
    hold on
    h4 = plot(SNRsdB, mean_dCth(:,:,k),...
        'Color', colors(color_idx, :), ...
        'LineStyle', line_styles{line_style_idx}, ...
        'Marker', markers{marker_idx}, ...
        'MarkerFaceColor', colors(color_idx, :),...
        'MarkerEdgeColor', colors(color_idx, :),...
        'LineWidth', k*0.5,...
        'DisplayName', leg_keys(k));
    
    set(gca, "YScale", 'log')
    xlabel('SNR [dB]')
    ylabel('$<|\cos(\hat{\theta})-\cos(\theta)|>~[deg]$')
    grid on
    axis square
    box on

    % Utiliser la première courbe ajoutée pour chaque subplot comme entrée pour la légende
    legendEntries = [legendEntries, h1];  % Ajoute la courbe réelle dans la légende
end

% Ajouter une légende commune
lgd = legend(legendEntries, 'Location', 'southoutside', 'Orientation', 'horizontal');
lgd.Layout.Tile = 'south'; 
legend boxoff

set(gcf, 'Position', [0.0198, 0.0009, 0.5255, 0.8824])
