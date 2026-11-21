%% STILL ERRORS- Needs to have data smoothly go from one to another rather than reload files, and error in matrix support

%% 1. Load data
eeglab nogui;
EEG = pop_loadset('filename','sub-G01S01_task-pddecision_eeg.set','filepath','C:\Users\cunni\Downloads\');

%% 2. MSPCA settings
level = 4;          
wname = 'db4';

%% 3. Run MSPCA on each epoch
EEG_clean = EEG;
EEG_clean.data = zeros(size(EEG.data), 'single');

for k = 1:EEG.trials
    Xk = double(EEG.data(:,:,k))';            
    N  = size(Xk,1);

    n  = ceil(N / 2^level) * 2^level;
    Xp = [Xk; repmat(Xk(end,:), n-N, 1)];

    [Xc, qual, npc] = wmspca(Xp, level, wname, 'kais');
    EEG_clean.data(:,:,k) = single(Xc(1:N,:)');   % drop padding
end

EEG_clean = eeg_checkset(EEG_clean);

%% 4. Simple display: one epoch, one channel
ch = 1; trial = 1;
t  = EEG.times / 1000;                        % seconds

figure;
subplot(2,1,1);
plot(t, squeeze(EEG.data(ch,:,trial)), 'k'); hold on;
plot(t, squeeze(EEG_clean.data(ch,:,trial)), 'r');
legend('Original','MSPCA'); xlabel('Time (s)'); ylabel('\muV');
title(sprintf('Channel %d, epoch %d', ch, trial));

subplot(2,1,2);
plot(t, squeeze(EEG.data(ch,:,trial) - EEG_clean.data(ch,:,trial)), 'b');
xlabel('Time (s)'); title('Removed component');

%% 5. Display: average across epochs (all-epoch comparison)
figure;
plot(t, mean(EEG.data(ch,:,:), 3), 'k'); hold on;
plot(t, mean(EEG_clean.data(ch,:,:), 3), 'r');
legend('Original ERP','MSPCA ERP'); xlabel('Time (s)'); ylabel('\muV');
title(sprintf('Channel %d, average over %d epochs', ch, EEG.trials));

%% 6. Optional: scroll through epochs in EEGLAB
pop_eegplot(EEG,       1, 1, 0, [], 'title', 'Original');
pop_eegplot(EEG_clean, 1, 1, 0, [], 'title', 'MSPCA cleaned');
%/////////////////////////////////////////////////////////////////////////////////////

% prototype_svm_lda
% One rest recording -> filter, correlation, LDA, and SVM.
% File: sub-G01S01_task-pdrest_eeg.set  (person S01 in triad G01, rest only)
% Run from this folder in MATLAB:  prototype_svm_lda

% I do not have the Signal Processing or Statistics toolboxes,
% so the filter, LDA, and SVM are the short formulas those toolboxes wrap.
%that is part of the reason the code is so long..... 

% Section map:
%   1. Setup folders
%   2. Load the .set file and print what is inside it
%   3. Z-score and bandpass (1-40 Hz)
%   4. Alpha-band correlation between electrodes
%   5. Build 2-second window features and minute-1 vs minute-3 labels
%   6. Run LDA and SVM (5-fold) and print weights
%   7. Save figures and results.txt
% Local functions below the script do the actual math.

%% 1. Setup folders
clear; clc; close all;
root = fileparts(mfilename('fullpath'));
if isempty(root), root = pwd; end
cd(root);
outDir = fullfile(root, 'prototype_figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% 2. Load the recording and report what is in it
% A .set file here is a MATLAB MAT-file. The extension is not .mat, so load
% needs the '-mat' flag or it tries to read the bytes as a text table.
S = load('-mat', fullfile(root, 'sub-G01S01_task-pddecision_eeg.set'));
if isfield(S, 'EEG'), S = S.EEG; end

fs = S.srate;
raw = double(S.data);
labels = {S.chanlocs.labels};
nChan = size(raw, 1);
nSamp = size(raw, 2);

fprintf('\n=== What is inside sub-G01S01_task-pdrest_eeg.set ===\n');
fprintf('Set name: %s\n', S.setname);
fprintf('Channels: %d   Samples: %d   Rate: %.0f Hz   Duration: %.0f s\n', ...
    nChan, nSamp, fs, nSamp / fs);
fprintf('Reference field: %s    Trials stored in the file: %.0f (continuous)\n', ...
    char(S.ref), S.trials);
fprintf('Channel order: %s\n', strjoin(labels, ', '));
fprintf('Stored amplitude range: %.3g to %.3g (NOT plausible microvolts)\n', ...
    min(raw(:)), max(raw(:)));
fprintf('Each channel is z-scored before filtering so the scale cannot dominate.\n');
fprintf('Events are not in this .set. The events file says 3 rest blocks of 60 s.\n\n');

%% 3. Put every channel on the same scale, then keep 1-40 Hz
% mean(raw, 2) is the mean of each channel across time. Dividing by std
% makes every channel have mean 0 and standard deviation 1. A channel stored
% as hundreds of millions can no longer drown the others. Correlation does
% not care about this step; band power does.
% z-score each channel across time (0 = normalize by N-1)
z = (raw - mean(raw, 2)) ./ std(raw, 0, 2);
% drop slow drift below 1 Hz and fast noise above 40 Hz
z = bandpass_fft(z, fs, 1, 40);

%% 4. Alpha correlation: which electrodes move together
% Second filter keeps the resting alpha rhythm (8-13 Hz). corrcoef wants
% each variable in a column, so alpha (channels x time) is transposed.
% The diagonal of R is always 1 (a channel with itself). The upper triangle
% is each unique pair once.
% keep only the alpha band
alpha = bandpass_fft(z, fs, 8, 13);
% 19 x 19 correlation matrix
R = corrcoef(alpha');
% pull the upper triangle, skipping the diagonal of 1s
pair = R(triu(true(nChan), 1));
% same upper-triangle mask, used to recover channel indexes
mask = triu(true(nChan), 1);
% ii(k), jj(k) are the two channel indexes of pair(k)
[ii, jj] = find(mask);
% largest correlation and which pair it belongs to
[best, kBest] = max(pair);
fprintf('Alpha (8-13 Hz) correlation across electrode pairs:\n');
fprintf('  mean %.2f   min %.2f   max %.2f\n', mean(pair), min(pair), max(pair));
fprintf('  strongest pair: %s-%s = %.2f\n\n', labels{ii(kBest)}, labels{jj(kBest)}, best);

%% 5. Features and labels
% rest_features cuts the 180 s into 2 s windows and returns 6 numbers per
% window: log power of theta, alpha, and beta, each averaged over frontal
% channels and over posterior channels. This one file can answer "does
% minute 1 of rest differ from minute 3?" Coupling between two people needs
% a second recording. Minute 2 is left out so the two classes are separated
% in time and have the same number of windows.
% Xall is windows x 6; t is the start time of each window, in seconds
[Xall, t] = rest_features(z, fs);
% keep minute 1 and minute 3; drop the middle minute
use = (t < 60) | (t >= 120);
% feature matrix the classifiers actually see (60 x 6)
X = Xall(use, :);
% label 0 = minute 1, label 1 = minute 3
y = double(t(use) >= 120);
featNames = {'frontal theta','posterior theta','frontal alpha', ...
    'posterior alpha','frontal beta','posterior beta'};

fprintf('Windows: %d in minute 1, %d in minute 3. Features: %d.\n', ...
    sum(y == 0), sum(y == 1), size(X, 2));
fprintf('Chance accuracy if the classifier only guesses: 50%%.\n');
fprintf('Mean posterior theta log-power: minute 1 = %.2f, minute 3 = %.2f\n', ...
    mean(X(y == 0, 2)), mean(X(y == 1, 2)));
fprintf('Mean posterior alpha log-power: minute 1 = %.2f, minute 3 = %.2f\n', ...
    mean(X(y == 0, 4)), mean(X(y == 1, 4)));

%% 6. Classify minute 1 vs minute 3
% freeze the random shuffle so a second run prints the same accuracies
rng(1);
% each window gets a fold id from 1 to 5, balanced across the two classes
fold = fold_of(y, 5);
% 5-fold LDA accuracy, plus weights fit on all 60 windows
[accLda, wLda] = lda_rest(X, y, fold);
% same split for the linear SVM, so the two accuracies are comparable
[accSvm, wSvm] = svm_rest(X, y, fold);

fprintf('\nWeights (positive points toward minute 3):\n');
fprintf('%-18s %10s %10s\n', 'feature', 'LDA', 'SVM');
% one row per feature
for k = 1:numel(featNames)
    % positive weight votes for minute 3
    fprintf('%-18s %+10.2f %+10.2f\n', featNames{k}, wLda(k), wSvm(k));
end

%% 7. Save the graphs and a text copy of the numbers
save_figures(z, fs, labels, R, X, y, accLda, accSvm, featNames, outDir);
write_results(outDir, accLda, accSvm, best, labels{ii(kBest)}, labels{jj(kBest)}, ...
    mean(pair), wLda, wSvm, featNames, ...
    mean(X(y == 0, 2)), mean(X(y == 1, 2)), mean(X(y == 0, 4)), mean(X(y == 1, 4)));
fprintf('\nFigures saved in:\n  %s\n', outDir);

%% ------------------------------------------------------------------------
function [acc, w] = lda_rest(X, y, fold)
% Linear discriminant analysis, 5-fold.
% Each pass trains on 4 folds and scores the held-out fold. The returned
% weight vector is a separate fit on every window, for interpretation only.
% It is not the model that produced acc.
% one test accuracy per fold
accs = zeros(5, 1);
% fold i is the test set this pass
for i = 1:5
    % logical masks: train on the other folds, test on fold i
    tr = fold ~= i; te = fold == i;
    % mean and std of each feature, from training windows only
    [mu, sd] = zstats(X(tr, :));
    % LDA predictions on the test fold
    pred = fisher(applyz(X(tr, :), mu, sd), y(tr), applyz(X(te, :), mu, sd));
    % fraction of test windows labeled correctly
    accs(i) = mean(pred == y(te));
end
% headline accuracy is the average of the five folds
acc = mean(accs);
% scale using all windows, only for the printed weights
[mu, sd] = zstats(X);
% direction from minute 1 toward minute 3
w = fisher_w(applyz(X, mu, sd), y);
fprintf('LDA 5-fold accuracy: %.1f%%   folds [%s]%%\n', ...
    100 * acc, strjoin(compose('%.0f', 100 * accs'), ', '));
end

function [acc, w] = svm_rest(X, y, fold)
% Linear support vector machine, same 5 folds as LDA.
% LDA aims the line using class averages. The SVM places the line so the
% empty gap between the two classes is as wide as possible.
% one test accuracy per fold
accs = zeros(5, 1);
for i = 1:5
    % same train/test split as lda_rest
    tr = fold ~= i; te = fold == i;
    % training-fold scale, so the test fold does not set the units
    [mu, sd] = zstats(X(tr, :));
    pred = linear_svm(applyz(X(tr, :), mu, sd), y(tr), applyz(X(te, :), mu, sd));
    % fraction correct on the held-out fold
    accs(i) = mean(pred == y(te));
end
acc = mean(accs);
% scale for the interpretation weights, using all windows
[mu, sd] = zstats(X);
% 7 weights: 6 features plus a bias in the last entry
w = linear_svm_w(applyz(X, mu, sd), y);
fprintf('SVM 5-fold accuracy: %.1f%%   folds [%s]%%\n', ...
    100 * acc, strjoin(compose('%.0f', 100 * accs'), ', '));
end

function pred = fisher(Xtr, ytr, Xte)
% Return only the class predictions. The weight vector is discarded here
% because the cross-validation loop only needs right/wrong.
[pred, ~] = fisher_w_predict(Xtr, ytr, Xte);
end

function w = fisher_w(Xtr, ytr)
% Return only the weight vector, scoring the training rows just to reuse
% the same formula as the prediction path.
[~, w] = fisher_w_predict(Xtr, ytr, Xtr);
end

function [pred, w] = fisher_w_predict(Xtr, ytr, Xte)
% Fisher LDA for two classes.
% mu0, mu1: average feature vector of minute 1 and minute 3.
% S: pooled covariance (how the 6 features vary together).
% w: direction from mu0 to mu1, divided by that covariance so a noisy
%    feature counts less. Solve S*w = (mu1-mu0)'.
% A test window is minute 3 when its projection passes the midpoint.
% 1 x 6 mean of minute-1 training windows
mu0 = mean(Xtr(ytr == 0, :), 1);
% 1 x 6 mean of minute-3 training windows
mu1 = mean(Xtr(ytr == 1, :), 1);
% how many windows are in each class
n0 = sum(ytr == 0); n1 = sum(ytr == 1);
% scatter of each class, added
S = cov(Xtr(ytr == 0, :)) * (n0 - 1) + cov(Xtr(ytr == 1, :)) * (n1 - 1);
% pooled covariance, plus a ridge so the solve cannot blow up
S = S / (n0 + n1 - 2) + 1e-5 * eye(size(Xtr, 2));
% weight vector, 6 x 1, pointing toward minute 3
w = S \ (mu1 - mu0)';
% threshold halfway between the two class means, in the w direction
thr = 0.5 * (mu0 + mu1) * w;
% 1 if the window is past the midpoint (minute 3), else 0
pred = double(Xte * w > thr);
end

function pred = linear_svm(Xtr, ytr, Xte)
% Fit the line on the training windows, then score the test windows.
% weights plus bias
w = linear_svm_w(Xtr, ytr);
% add a column of ones so the last weight acts as the intercept
score = [Xte, ones(size(Xte, 1), 1)] * w;
% positive score means minute 3 (the +1 class)
pred = double(score > 0);
end

function w = linear_svm_w(Xtr, ytr)
% Pegasos linear SVM.
% Labels become -1 (minute 1) and +1 (minute 3). A column of ones is added
% so the last weight is the bias. For 30 passes over the training windows:
%   - if the window is already outside the margin on the correct side,
%     shrink the feature weights a little (keeps the line simple);
%   - if it is inside the margin or on the wrong side, step the weights
%     toward classifying that window correctly.
% Step size eta shrinks as more windows are seen. lambda sets how hard
% the weights are pulled toward zero.
% 0/1 labels become -1/+1
yy = 2 * ytr - 1;
% extra column of ones = intercept
X = [Xtr, ones(size(Xtr, 1), 1)];
% n windows, d = 7 (6 features + bias)
[n, d] = size(X);
% start from a line through the origin
w = zeros(d, 1);
% regularization strength
lambda = 0.01;
% step counter; the learning rate falls as t grows
t = 0;
% 30 full passes through the training windows
for epoch = 1:30
    % visit windows in order, so the fit is repeatable
    for i = 1:n
        % global step index
        t = t + 1;
        % learning rate for this step
        eta = 1 / (lambda * t);
        % margin violation: wrong side, or correct but too close to the line
        if yy(i) * (X(i, :) * w) < 1
            % shrink, then push toward the correct class
            w = (1 - eta * lambda) * w + eta * yy(i) * X(i, :)';
        else
            % already correct: shrink feature weights, leave the bias alone
            w(1:end-1) = (1 - eta * lambda) * w(1:end-1);
        end
    end
end
end

function [X, t] = rest_features(z, fs)
% Turn the continuous recording into one row per 2-second window.
% Columns, in order: frontal theta, posterior theta, frontal alpha,
% posterior alpha, frontal beta, posterior beta. Power is the mean of the
% squared samples. log() compresses it. Values are negative because one
% band is only a fraction of the z-scored signal; closer to zero means
% more power. Channel indexes assume the order in this specific file.
% row 1 theta, row 2 alpha, row 3 beta, in Hz
bands = [4 8; 8 13; 13 30];
% Fp1 Fp2 Fz F3 F4, indexes into this file's channel order
frontal = [10 11 4 3 5];
% O1 O2 Pz P3 P4
posterior = [14 15 9 1 7];
% samples in one 2-second window (600 at 300 Hz)
win = 2 * fs;
% how many full windows fit (90 in 180 s); a leftover tail is dropped
nwin = floor(size(z, 2) / win);
% preallocate the feature matrix
X = zeros(nwin, 6);
% one pass per frequency band
for b = 1:3
    % whole recording, this band only
    xb = bandpass_fft(z, fs, bands(b, 1), bands(b, 2));
    % each 2-second chunk
    for widx = 1:nwin
        % power of every channel in this window
        p = mean(xb(:, (widx - 1) * win + (1:win)) .^ 2, 2);
        % odd column: mean frontal power, then log
        X(widx, 2 * b - 1) = log(mean(p(frontal)));
        % even column: mean posterior power, then log
        X(widx, 2 * b) = log(mean(p(posterior)));
    end
end
% start time of each window, in seconds: 0, 2, 4, ...
t = (0:nwin - 1)' * 2;
end

function y = bandpass_fft(x, fs, lo, hi)
% Brick-wall bandpass. x is channels x time.
% Take the FFT of each channel, zero every bin outside [lo, hi], and
% transform back. The mirror band just below the sampling rate is kept
% too, because a real signal has matching negative frequencies. real()
% drops tiny imaginary leftovers from rounding.
% number of time samples
n = size(x, 2);
% frequency of each FFT bin, in Hz
f = (0:n - 1) * (fs / n);
% positive band and its mirror
keep = (f >= lo & f <= hi) | (f >= (fs - hi) & f <= (fs - lo));
% FFT along time, one channel per row
Y = fft(x, [], 2);
% delete frequencies outside the band
Y(:, ~keep) = 0;
% back to time; real() removes rounding noise
y = real(ifft(Y, [], 2));
end

function fold = fold_of(y, k)
% Assign each window to a fold from 1 to k.
% The two classes are shuffled separately, then dealt out round-robin, so
% every fold has the same mix of minute 1 and minute 3. Call rng first if
% you need the shuffle to be repeatable.
% one fold id per window, same shape as y
fold = zeros(size(y));
% do minute 1, then minute 3
for c = 0:1
    % indexes of windows in this class
    w = find(y == c);
    % shuffle those indexes
    w = w(randperm(numel(w)));
    % deal them into folds 1..k, then repeat
    fold(w) = mod((0:numel(w) - 1)', k) + 1;
end
end

function [mu, sd] = zstats(X)
% Column mean and standard deviation, used to z-score features inside a fold.
% 1 x features
mu = mean(X, 1);
% 1 x features, normalized by N-1
sd = std(X, 0, 1);
% a constant column would divide by zero; leave it unchanged
sd(sd == 0) = 1;
end

function Z = applyz(X, mu, sd)
% Apply a previously computed mean and std. Training and test windows must
% share the training stats, otherwise the test set leaks into the scale.
Z = (X - mu) ./ sd;
end

function save_figures(z, fs, labels, R, X, y, accLda, accSvm, featNames, outDir)
% Write three PNGs. Figures are hidden so the script can run with -batch.
t = (0:size(z, 2) - 1) / fs;
show = [4 8 14];
span = t <= 10;

% Figure 1: filtered traces.
f = figure('Visible', 'off', 'Position', [80 80 980 620]);
for k = 1:3
    subplot(3, 1, k);
    plot(t(span), z(show(k), span), 'Color', [0.1 0.25 0.55]);
    ylabel(labels{show(k)});
    xlim([0 10]); grid on;
end
xlabel('Time (s)');
sgtitle('Filtered rest EEG, first 10 s (each channel z-scored, 1-40 Hz)');
exportgraphics(f, fullfile(outDir, '01_filtered_traces.png'), 'Resolution', 140);

% Figure 2: alpha correlation matrix. Color scale is fixed at -1 to +1
% so a later recording can be compared with this one.
f = figure('Visible', 'off', 'Position', [80 80 760 640]);
imagesc(R, [-1 1]); axis square; colormap(parula); colorbar;
set(gca, 'XTick', 1:numel(labels), 'YTick', 1:numel(labels), ...
    'XTickLabel', labels, 'YTickLabel', labels, 'TickLength', [0 0]);
xtickangle(45);
title('Alpha-band correlation between electrodes (one person)');
exportgraphics(f, fullfile(outDir, '02_alpha_correlation.png'), 'Resolution', 140);

% Figure 3, left: only two of the six features, so the class split is visible.
% The accuracy bars on the right use all six features.
f = figure('Visible', 'off', 'Position', [80 80 980 460]);
subplot(1, 2, 1);
hold on;
scatter(X(y == 0, 2), X(y == 0, 4), 28, [0.2 0.45 0.75], 'filled');
scatter(X(y == 1, 2), X(y == 1, 4), 28, [0.85 0.35 0.15], 'filled');
xlabel(featNames{2}); ylabel(featNames{4});
legend({'Minute 1', 'Minute 3'}, 'Location', 'best');
title('Two of the six features'); grid on; hold off;

subplot(1, 2, 2);
bar([50, 100 * accLda, 100 * accSvm]);
set(gca, 'XTickLabel', {'Chance', 'LDA', 'SVM'});
ylabel('Accuracy (%)'); ylim([0 100]);
title('Minute 1 vs minute 3 of rest'); grid on;
exportgraphics(f, fullfile(outDir, '03_lda_svm_results.png'), 'Resolution', 140);
close all;
end

function write_results(outDir, accLda, accSvm, best, aName, bName, meanR, wLda, wSvm, featNames, th1, th3, a1, a3)
% Text copy of the numbers, so the presentation does not depend on the command window.
% th1/th3 are posterior theta log-power in minute 1 and minute 3.
% a1/a3 are the same for posterior alpha. aName-bName is the strongest pair.
fid = fopen(fullfile(outDir, 'results.txt'), 'w');
fprintf(fid, 'LDA accuracy: %.1f%%\n', 100 * accLda);
fprintf(fid, 'SVM accuracy: %.1f%%\n', 100 * accSvm);
fprintf(fid, 'Chance: 50%%\n');
fprintf(fid, 'Mean alpha correlation: %.2f\n', meanR);
fprintf(fid, 'Strongest alpha pair: %s-%s = %.2f\n', aName, bName, best);
fprintf(fid, 'Posterior theta log-power minute 1 / 3: %.2f / %.2f\n', th1, th3);
fprintf(fid, 'Posterior alpha log-power minute 1 / 3: %.2f / %.2f\n', a1, a3);
% one line of weights per feature
for k = 1:numel(featNames)
    fprintf(fid, '%s  LDA %+.2f  SVM %+.2f\n', featNames{k}, wLda(k), wSvm(k));
end
fclose(fid);
end

%/////////////////////////////////////////////////////////////////////////////////////
% GEDAI Before-and-After Comparison
% Compares the original EEG signal with the GEDAI-cleaned signal.

% Select original and cleaned datasets
EEG_original = ALLEEG(1);
EEG_cleaned = ALLEEG(2);

% Select channel to display
channel_name = 'Fz';

% Find the channel index in each dataset
original_channel = find(strcmpi({EEG_original.chanlocs.labels}, channel_name), 1);
cleaned_channel = find(strcmpi({EEG_cleaned.chanlocs.labels}, channel_name), 1);

% Check that the channel exists in both datasets
if isempty(original_channel) || isempty(cleaned_channel)
    error('The selected channel was not found in both datasets.');
end

% Select the first 10 seconds
duration = 10;
num_samples = round(duration * EEG_original.srate);
time = (0:num_samples-1) / EEG_original.srate;

% Extract original and cleaned signals
original_signal = double(EEG_original.data(original_channel, 1:num_samples));
cleaned_signal = double(EEG_cleaned.data(cleaned_channel, 1:num_samples));

% Calculate shared Y-axis limits using both signals
all_signal_values = [original_signal(:); cleaned_signal(:)];

y_min = min(all_signal_values);
y_max = max(all_signal_values);

% Add 10% padding to the shared axis range
padding = 0.10 * (y_max - y_min);

% Handle the case where both signals have identical values
if padding == 0
    padding = 1;
end

shared_ylim = [y_min - padding, y_max + padding];

% Create output folder
output_folder = fullfile(pwd, 'GEDAI_Images');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Plot original signal
figure('Color', 'white');

plot(time, original_signal, 'r', 'LineWidth', 0.8);

xlabel('Time (seconds)');
ylabel('Amplitude (dataset units)');
title('Original EEG Signal - Fz');
grid on;
xlim([0 duration]);
ylim(shared_ylim);

exportgraphics(gcf, ...
    fullfile(output_folder, 'EEG_Before_GEDAI.png'), ...
    'Resolution', 300);

% Plot cleaned signal
figure('Color', 'white');

plot(time, cleaned_signal, 'b', 'LineWidth', 0.8);

xlabel('Time (seconds)');
ylabel('Amplitude (dataset units)');
title('GEDAI-Cleaned EEG Signal - Fz');
grid on;
xlim([0 duration]);
ylim(shared_ylim);

exportgraphics(gcf, ...
    fullfile(output_folder, 'EEG_After_GEDAI.png'), ...
    'Resolution', 300);

disp('Before-and-after images successfully exported.');
