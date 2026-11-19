%% 1. Load data
eeglab nogui;
EEG = pop_loadset('filename','sub-G01S01_task-pddecision_eeg.set','filepath','/Users/elo/Documents/University/QMind/EEG/full_eeg_data_set/ds007822/sub-G01S01/eeg/');

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