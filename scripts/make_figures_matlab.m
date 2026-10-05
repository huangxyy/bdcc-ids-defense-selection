function make_experiment_figures_matlab(outputDir)
%MAKE_EXPERIMENT_FIGURES_MATLAB
% Regenerate selected experiment figures with a consistent journal style.
%
% Outputs:
%   figures/Fig2_pareto_v2.png/.pdf
%   figures/Fig3_theta_v2.png/.pdf
%   figures/Fig5_risk_surface_v2.png/.pdf

if nargin < 1 || isempty(outputDir)
    here = fileparts(mfilename('fullpath'));
    outputDir = fullfile(here, 'figures');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

C = palette();
[defenses, backbones, risk, pareto, admissible] = riskData();

makeFig2(outputDir, defenses, backbones, risk, pareto, admissible, C);
makeFig3(outputDir, defenses, backbones, risk, pareto, C);
makeFig5(outputDir, defenses, backbones, C);
end

% ======================================================================
% Fig. 2: Pareto front comparison
% ======================================================================

function makeFig2(outputDir, defenses, backbones, risk, pareto, admissible, C)
fig = figure('Color', C.white, 'Units', 'inches', ...
    'Position', [0.5 0.5 7.2 3.20], ...
    'PaperUnits', 'inches', 'PaperPosition', [0 0 7.2 3.20], ...
    'PaperSize', [7.2 3.20], 'Renderer', 'painters');
tl = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

fairAll = reshape(risk(:,4,:), [], 1);
fairLim = [min(fairAll) max(fairAll)];
sizeMin = 95;
sizeMax = 310;
cmap = fairnessMap(256);

for b = 1:3
    ax = nexttile(tl, b);
    values = risk(:,:,b);
    hold(ax, 'on');
    styleAxes(ax, C);
    title(ax, panelTitle(b, backbones{b}), 'FontName', 'Arial', ...
        'FontWeight', 'bold', 'FontSize', 10.6, 'Color', C.ink);

    x = values(:,1);
    y = values(:,2);
    cost = values(:,3);
    fair = values(:,4);
    markerSize = sizeMin + (sizeMax - sizeMin) .* normalize01(cost);

    for i = 1:numel(defenses)
        if ~admissible(i,b)
            edgeColor = C.muted;
            edgeWidth = 1.2;
            markerShape = 'o';
            faceColor = C.white;
        elseif pareto(i,b)
            edgeColor = C.ink;
            edgeWidth = 1.0;
            markerShape = 'o';
            faceColor = 'flat';
        else
            edgeColor = C.dominated;
            edgeWidth = 1.9;
            markerShape = 'd';
            faceColor = 'flat';
        end
        scatter(ax, x(i), y(i), markerSize(i), fair(i), ...
            'Marker', markerShape, ...
            'MarkerFaceColor', faceColor, ...
            'MarkerEdgeColor', edgeColor, ...
            'LineWidth', edgeWidth, ...
            'MarkerFaceAlpha', 0.94, ...
            'MarkerEdgeAlpha', 1.0);

        if admissible(i,b) && fair(i) > 0.40
            labelColor = C.white;
        else
            labelColor = C.ink;
        end
        text(ax, x(i), y(i), abbrev(defenses{i}), ...
            'FontName', 'Arial', 'FontSize', 7.7, 'FontWeight', 'bold', ...
            'Color', labelColor, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'Clipping', 'on');
    end

    dominatedNames = defenses(~pareto(:,b) & admissible(:,b));
    inadmissibleNames = defenses(~admissible(:,b));
    parts = {};
    if ~isempty(inadmissibleNames)
        parts{end+1} = ['inadmissible: ' strjoin(inadmissibleNames, ', ')];
    end
    if ~isempty(dominatedNames)
        parts{end+1} = ['dominated: ' strjoin(dominatedNames, ', ')];
    end
    if isempty(parts)
        parts = {'none dominated'};
    end
    text(ax, 0.02, 0.04, sprintf('|P|=%d/6; %s', sum(pareto(:,b)), strjoin(parts, '; ')), ...
        'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 7.2, ...
        'Color', C.muted, 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom');

    xlabel(ax, '\phi_1 Clean F_1', 'FontName', 'Arial', 'FontSize', 9.6, 'Color', C.ink);
    if b == 1
        ylabel(ax, '\phi_2 Resilience', 'FontName', 'Arial', 'FontSize', 9.6, 'Color', C.ink);
    else
        ylabel(ax, '');
        ax.YTickLabel = {};
    end
    xPad = max(0.004, 0.08 * (max(x) - min(x)));
    yPad = max(0.004, 0.08 * (max(y) - min(y)));
    xlim(ax, [min(x) - xPad, max(x) + xPad]);
    ylim(ax, [min(y) - yPad, max(y) + yPad]);
    colormap(ax, cmap);
    caxis(ax, fairLim);
end

exportPair(fig, fullfile(outputDir, 'Fig2_pareto_v2'));
close(fig);
end

% ======================================================================
% Fig. 3: theta sensitivity
% ======================================================================

function makeFig3(outputDir, defenses, backbones, risk, pareto, C)
thetaRob = 0.00:0.02:0.85;
thetaCost = 0.00:0.02:0.85;
fig = figure('Color', C.white, 'Units', 'inches', ...
    'Position', [0.5 0.5 7.2 3.45], ...
    'PaperUnits', 'inches', 'PaperPosition', [0 0 7.2 3.45], ...
    'PaperSize', [7.2 3.45], 'Renderer', 'painters');
tl = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for b = 1:3
    ax = nexttile(tl, b);
    hold(ax, 'on');
    styleAxes(ax, C);
    title(ax, panelTitle(b, backbones{b}), 'FontName', 'Arial', ...
        'FontWeight', 'bold', 'FontSize', 10.6, 'Color', C.ink);

    values = risk(:,:,b);
    front = pareto(:,b);
    normVals = nan(size(values));
    for k = 1:4
        v = values(front,k);
        lo = min(v);
        hi = max(v);
        if abs(hi - lo) < eps
            normVals(front,k) = 0;
        else
            normVals(front,k) = (values(front,k) - lo) ./ (hi - lo);
        end
    end

    winnerGrid = nan(numel(thetaCost), numel(thetaRob));
    for r = 1:numel(thetaCost)
        for c = 1:numel(thetaRob)
            w2 = thetaRob(c);
            w3 = thetaCost(r);
            rem = 1 - w2 - w3;
            if rem < -1e-12
                continue;
            end
            w = [rem/2 w2 w3 rem/2];
            scores = nan(numel(defenses), 1);
            for i = 1:numel(defenses)
                if front(i)
                    scores(i) = sum(w .* normVals(i,:));
                end
            end
            [~, winnerGrid(r,c)] = max(scores, [], 'omitnan');
        end
    end

    h = imagesc(ax, thetaRob, thetaCost, winnerGrid);
    h.AlphaData = ~isnan(winnerGrid);
    colormap(ax, C.colors);
    caxis(ax, [1 6]);
    ax.YDir = 'normal';
    plot(ax, [0 0.85], [1 0.15], '--', 'Color', C.rule2, 'LineWidth', 1.1);
    labelPreferenceRegions(ax, thetaRob, thetaCost, winnerGrid, defenses, C);

    xlabel(ax, '\theta_2 robustness weight', 'FontName', 'Arial', 'FontSize', 9.4, 'Color', C.ink);
    if b == 1
        ylabel(ax, '\theta_3 cost weight', 'FontName', 'Arial', 'FontSize', 9.4, 'Color', C.ink);
    else
        ylabel(ax, '');
        ax.YTickLabel = {};
    end
    xlim(ax, [0 0.85]);
    ylim(ax, [0 0.85]);
    axis(ax, 'square');
end

legAx = axes(fig, 'Position', [0.14 0.020 0.76 0.095], 'Visible', 'off');
hold(legAx, 'on');
legAx.XLim = [0 1];
legAx.YLim = [0 1];
legendX = [0.08 0.41 0.74 0.08 0.41 0.74];
legendY = [0.70 0.70 0.70 0.28 0.28 0.28];
for i = 1:numel(defenses)
    plot(legAx, [legendX(i)-0.050 legendX(i)+0.020], [legendY(i) legendY(i)], ...
        '-', 'Color', C.colors(i,:), 'LineWidth', 2.6);
    text(legAx, legendX(i)+0.035, legendY(i), defenses{i}, ...
        'FontName', 'Arial', 'FontSize', 7.8, 'Color', C.ink, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end

exportPair(fig, fullfile(outputDir, 'Fig3_theta_v2'));
close(fig);
end

% ======================================================================
% Fig. 5: risk-surface heatmap
% ======================================================================

function makeFig5(outputDir, defenses, backbones, C)
scenarios = {'FGSM\epsilon=.05','PGD\epsilon=.10','C&W L_2','APGD\epsilon=.10','Mask-PGD\epsilon=.10'};

% NOTE: the values below are the submitted-manuscript snapshot and have not
% been regenerated from the revision CSVs. Refresh them (or use
% outputs/figures/risk_surface_heatmap.png, which is generated from the
% revision outputs) before reusing this figure.
riskSurf = zeros(6, 5, 3);
riskSurf(:,:,1) = [ ...
    0.82 0.76 0.50 0.74 0.76
    0.88 0.86 0.73 0.85 0.86
    0.86 0.83 0.64 0.81 0.83
    0.89 0.88 0.79 0.87 0.88
    0.86 0.82 0.65 0.81 0.82
    0.88 0.84 0.67 0.82 0.84];
riskSurf(:,:,2) = [ ...
    0.74 0.54 0.01 0.50 0.54
    0.87 0.80 0.39 0.79 0.80
    0.82 0.70 0.13 0.66 0.70
    0.86 0.80 0.58 0.78 0.80
    0.81 0.63 0.03 0.58 0.63
    0.90 0.83 0.28 0.81 0.83];
riskSurf(:,:,3) = [ ...
    0.84 0.79 0.64 0.77 0.78
    0.89 0.87 0.77 0.87 0.87
    0.88 0.86 0.74 0.86 0.86
    0.91 0.89 0.76 0.88 0.89
    0.87 0.83 0.70 0.83 0.83
    0.89 0.87 0.73 0.85 0.87];

fig = figure('Color', C.white, 'Units', 'inches', ...
    'Position', [0.5 0.5 7.2 3.35], ...
    'PaperUnits', 'inches', 'PaperPosition', [0 0 7.2 3.35], ...
    'PaperSize', [7.2 3.35], 'Renderer', 'painters');
tl = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
cmap = redYellowGreenMap(256);

for b = 1:3
    ax = nexttile(tl, b);
    imagesc(ax, riskSurf(:,:,b), [0 0.95]);
    colormap(ax, cmap);
    hold(ax, 'on');
    title(ax, panelTitle(b, backbones{b}), 'FontName', 'Arial', ...
        'FontWeight', 'bold', 'FontSize', 10.2, 'Color', C.ink);
    ax.FontName = 'Arial';
    ax.FontSize = 8.0;
    ax.TickLength = [0 0];
    ax.Box = 'off';
    ax.XTick = 1:numel(scenarios);
    ax.XTickLabel = scenarios;
    ax.XTickLabelRotation = 35;
    ax.YTick = 1:numel(defenses);
    if b == 1
        ax.YTickLabel = defenses;
    else
        ax.YTickLabel = {};
    end
    ax.YDir = 'normal';
    ax.XColor = C.ink;
    ax.YColor = C.ink;

    for x = 0.5:1:5.5
        plot(ax, [x x], [0.5 6.5], 'Color', C.white, 'LineWidth', 0.9);
    end
    for y = 0.5:1:6.5
        plot(ax, [0.5 5.5], [y y], 'Color', C.white, 'LineWidth', 0.9);
    end
    for r = 1:6
        for c = 1:5
            v = riskSurf(r,c,b);
            if v < 0.42 || v > 0.82
                txtColor = C.white;
            else
                txtColor = C.ink;
            end
            text(ax, c, r, sprintf('%.2f', v), ...
                'FontName', 'Arial', 'FontWeight', 'bold', ...
                'FontSize', 7.8, 'Color', txtColor, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        end
    end
    xlim(ax, [0.5 5.5]);
    ylim(ax, [0.5 6.5]);
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Robust F_1';
cb.Label.FontName = 'Arial';
cb.Label.FontSize = 9.2;
cb.FontName = 'Arial';
cb.FontSize = 8.0;
cb.Color = C.ink;

exportPair(fig, fullfile(outputDir, 'Fig5_risk_surface_v2'));
close(fig);
end

% ======================================================================
% Data and style helpers
% ======================================================================

function [defenses, backbones, risk, pareto, admissible] = riskData()
% Revision snapshot (official split, ten seeds, PGD alpha=eps/10, 50 steps,
% tau2=0.90, tau4=0.60). Regenerate from outputs/<backbone>/risk_profile_4d.csv
% before rebuilding the manuscript figures. Defenses are ordered as below:
% 1 StdTrain, 2 PGD-AT, 3 Constrained, 4 TRADES, 5 Free AT, 6 Class-Aware.
defenses = {'Standard','PGD-AT','Constrained','TRADES','Free AT','Class-Aware'};
backbones = {'MLP','1D-CNN','FT-Transformer'};

risk = zeros(6,4,3);
pareto = true(6,3);

risk(:,:,1) = [ ...
    0.8822 0.8418 1.0000 0.3647
    0.8564 0.9967 0.2083 0.9687
    0.8607 0.9637 0.2073 0.7904
    0.8557 0.9979 0.1658 0.9643
    0.8755 0.9209 0.3326 0.6045
    0.8568 0.9897 0.2039 0.9774];

risk(:,:,2) = [ ...
    0.8534 0.8097 1.0000 0.4934
    0.8278 0.9844 0.1836 0.9631
    0.8362 0.9435 0.1842 0.8522
    0.8236 0.9947 0.1554 0.9721
    0.8502 0.8971 0.3178 0.7666
    0.8291 0.9689 0.1853 0.9469];

risk(:,:,3) = [ ...
    0.8851 0.8921 1.0000 0.6687
    0.8567 0.9975 0.3334 0.9635
    0.8578 0.9934 0.3347 0.9534
    0.8542 0.9962 0.2936 0.9571
    0.8780 0.9431 0.3171 0.8259
    0.8561 0.9954 0.3316 0.9685];

% Admissibility floors remove StdTrain on every backbone and Free AT on the
% 1D-CNN (phi2 = 0.8971 < tau2). The Pareto flags are the Eq. 4 front of the
% remaining admissible candidates.
admissible = true(6,3);
admissible(1,:) = false;
admissible(5,2) = false;
pareto(:,1) = [false; true(5,1)];
pareto(:,2) = [false; true(3,1); false; true];
pareto(:,3) = [false; true(5,1)];
end

function C = palette()
C.white = [1 1 1];
C.ink = [22 30 38] / 255;
C.muted = [95 105 116] / 255;
C.rule = [225 229 234] / 255;
C.rule2 = [154 164 175] / 255;
C.dominated = [150 58 58] / 255;
C.colors = [ ...
    66 116 204
    232 126 48
    79 171 86
    200 74 82
    139 103 183
    143 112 65] / 255;
C.markers = {'o','s','d','^','v','p'};
end

function cmap = fairnessMap(n)
anchors = [ ...
    232 245 233
    165 214 167
    76 175 80
    27 94 32] / 255;
x = linspace(0, 1, size(anchors,1));
xi = linspace(0, 1, n);
cmap = zeros(n,3);
for k = 1:3
    cmap(:,k) = interp1(x, anchors(:,k), xi, 'pchip');
end
cmap(cmap < 0) = 0;
cmap(cmap > 1) = 1;
end

function [dx, dy, ha, va] = labelOffset(i, b)
dxMap = [ ...
    -0.0006  0.0006 -0.0004
    -0.0003  0.0006  0.0005
     0.0008 -0.0004 -0.0005
     0.0008 -0.0005  0.0004
    -0.0009  0.0007  0.0004
     0.0007  0.0005  0.0005];
dyMap = [ ...
    -0.011 -0.015 -0.012
     0.009  0.010  0.006
     0.007 -0.012 -0.008
     0.009  0.010  0.008
    -0.008 -0.014 -0.010
     0.009  0.008 -0.009];
dx = dxMap(i,b);
dy = dyMap(i,b);
ha = 'center';
va = 'middle';
end

function s = abbrev(name)
switch name
    case 'Standard'
        s = 'StdTrain';
    case 'PGD-AT'
        s = 'PGD';
    case 'Constrained'
        s = 'Cons';
    case 'TRADES'
        s = 'TRD';
    case 'Free AT'
        s = 'Free';
    case 'Class-Aware'
        s = 'CA';
    otherwise
        s = name;
end
end

function labelPreferenceRegions(ax, thetaRob, thetaCost, winnerGrid, defenses, C)
for i = 1:numel(defenses)
    [rr, cc] = find(winnerGrid == i);
    if numel(rr) < 18
        continue;
    end
    x = mean(thetaRob(cc));
    y = mean(thetaCost(rr));
    text(ax, x, y, abbrev(defenses{i}), ...
        'FontName', 'Arial', 'FontSize', 10.4, 'FontWeight', 'bold', ...
        'Color', C.white, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'Margin', 0.5);
end
end

function drawWinnerBands(ax, theta2, winner, C)
changes = [1 find(diff(winner) ~= 0) + 1 numel(winner) + 1];
for k = 1:(numel(changes)-1)
    first = changes(k);
    last = changes(k+1) - 1;
    x0 = theta2(first);
    if last < numel(theta2)
        x1 = theta2(last + 1);
    else
        x1 = theta2(last);
    end
    wi = winner(first);
    patch(ax, [x0 x1 x1 x0], [0.965 0.965 0.995 0.995], C.colors(wi,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.82);
    text(ax, (x0 + x1)/2, 0.95, abbrevColorLabel(wi), ...
        'FontName', 'Arial', 'FontSize', 6.7, 'Color', C.ink, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'Clipping', 'on');
end
text(ax, 0.102, 0.994, 'selected', 'FontName', 'Arial', 'FontSize', 6.6, ...
    'Color', C.muted, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
end

function s = abbrevColorLabel(idx)
labels = {'Std','PGD','Cons','TRD','Free','CA'};
s = labels{idx};
end

function out = winnerSegments(theta2, winner, defenses)
changes = [1 find(diff(winner) ~= 0) + 1 numel(winner) + 1];
parts = cell(1, numel(changes)-1);
for k = 1:(numel(changes)-1)
    first = changes(k);
    last = changes(k+1) - 1;
    x0 = theta2(first);
    if last < numel(theta2)
        x1 = theta2(last + 1);
    else
        x1 = theta2(last);
    end
    parts{k} = sprintf('%s %.2f-%.2f', abbrev(defenses{winner(first)}), x0, x1);
end
out = strjoin(parts, '; ');
end

function styleAxes(ax, C)
ax.FontName = 'Arial';
ax.FontSize = 8.3;
ax.Box = 'off';
ax.LineWidth = 0.8;
ax.XColor = C.ink;
ax.YColor = C.ink;
grid(ax, 'on');
ax.GridColor = C.rule;
ax.GridAlpha = 0.8;
try
    disableDefaultInteractivity(ax);
    ax.Toolbar.Visible = 'off';
catch
end
end

function out = panelTitle(idx, name)
letters = {'(a)','(b)','(c)'};
out = [letters{idx} ' ' name];
end

function y = normalize01(x)
lo = min(x);
hi = max(x);
if abs(hi - lo) < eps
    y = ones(size(x));
else
    y = (x - lo) ./ (hi - lo);
end
end

function cmap = redYellowGreenMap(n)
anchors = [ ...
    180 25 55
    236 93 55
    255 220 120
    194 226 112
    44 145 84] / 255;
x = linspace(0, 1, size(anchors,1));
xi = linspace(0, 1, n);
cmap = zeros(n,3);
for k = 1:3
    cmap(:,k) = interp1(x, anchors(:,k), xi, 'pchip');
end
cmap(cmap < 0) = 0;
cmap(cmap > 1) = 1;
end

function exportPair(fig, outputStem)
drawnow;
exportgraphics(fig, [outputStem '.png'], 'Resolution', 600, 'BackgroundColor', 'white');
exportgraphics(fig, [outputStem '.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'white');
fprintf('Saved %s.png and %s.pdf\n', outputStem, outputStem);
end
