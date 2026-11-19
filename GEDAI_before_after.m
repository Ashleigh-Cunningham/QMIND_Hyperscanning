
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
