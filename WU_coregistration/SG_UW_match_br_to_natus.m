%% Script used at the UW-Madison site for Cogitate Project
% for the purpose of cross-correlation of Natus clinical data
% with Blackrock data (containing triggers, and jumped neural channel)
% Script initially adapted from Saalmann Lab (https://saalmannlab.psych.wisc.edu/)
% and used by Sleep and Consciousness Lab (https://sleep-and-consciousness.wisc.edu/)
% Urszula version January 2024

%% Add to the path
% Make sure that the following are added to your Matlab path:
% Fieldtrip (https://www.fieldtriptoolbox.org/download.php, version 20220104 was used);
% EDFreadwrite (https://www.mathworks.com/matlabcentral/fileexchange/36530-read-write-edf-files);
% Blackrock script (https://github.com/BlackrockNeurotech/NPMK/releases);
% Neuralynx script (https://neuralynx.fh-co.com/research-software/#matlab-netcom-utilities)
% and custom loadNlxData function.
% Note that you can use Blackrock OR Neuralynx as two alternatives to record trigger data.

%% INPUTS - check parameters
homedir = '//';
subject = {'SG1XX'}; % subject name
prefix = 'SG1XX';
blackdir = (strcat(homedir, subject, '/Triggers/XXXXXXXXXX.ns3'));
PD_channel = 1; % insert the PD channel number
channel_br = 1; % insert channel of interest in blackrock data
subdir = strcat(homedir, subject, '/raw_edf/');
natus_file_name = char(strcat(subdir, 'PXXX_reduced.edf')); % file in EDF or .mat format 
channel_natus = 1; % channel of interest in natus data

% data of interest
check_all_data = 'yes'; % entire time window - 'yes', selected bins - 'no' 
toi_onset = 0.2; % hours since start of Natus data
toi_offset = 1.2; % hours since start of Natus data
window_ns = 30; % size of window (in seconds) to search for true match point on either side of high correlation points

% How many SDs above the mean correlation should the downsampled cross-correlation 
% coefficient be to be considered for alignment in native space?
sd_thresh = 4; % 5sd if using full dataset, 4sd if otherwise

% downsampling frequency
targ_sr = 512; % target sampling rate for correlation analysis
% If you are having trouble finding matches, try increasing the sampling rate. 
% (This is especially true if the natus data is sampled at 2048Hz.)

% inputs that most likely would not change
kerns = [50, 25, 10, 1]; % size of skips to try to perform initial alignment

% bandpass-filter settings
lpfreq = 200;
lpfiltord = 6;
padding = 2;
bsfiltord = 3;
bsfreq = [59 61; 119 121; 179 181];

%% LOAD Blackrock data
ns3file = char(blackdir);
openNSx(ns3file)
chan_br = double(NS3.Data(channel_br, :));
chan_PD = double(NS3.Data(PD_channel, :));

%% ALTERNATIVELY: load Neuralynx data
% % run conversion from NLX.
% CSC_FILE = 'CSC5_0XXX.ncs';
% [Timestamps, ChannelNumbers, SampleFrequencies, NumberOfValidSamples, Samples, Header] = Nlx2MatCSC(CSC_FILE, [1 1 1 1 1], 1, 1, [] );
% 
% fs = SampleFrequencies(1);
% chan_br = mean(Samples)';
% chan_PD = Timestamps';
% 
% [data_nlx, triggers, labels, fs] = loadNlxData('\\'); % insert data path
% chan_br = data_nlx(:,1)';
% chan_PD = triggers';

%% LOAD Natus data
cfg = [];
cfg.dataset = natus_file_name;
cfg.continuous = 'yes';
cfg.channel = 'all';
[n_data,n_header,n_cfg] = lab_read_edf(natus_file_name,cfg);
data = ft_preprocessing(cfg);

% multiply by -1 to flip the data
B=-1;
data.trial = cellfun(@(x) x * B, data.trial, 'UniformOutput', false);

%% Sampling freq details
hosp_sr = data.fsample; 
br_sr = NS3.MetaTags.SamplingFreq;

%% Chunking Natus data to time of interest for visual inspection
if strcmp(check_all_data, 'yes') % 'no' if from huge edf and I take specific time
    toi_onset_samples = 1;
    toi_offset_samples = size(data.trial{1, 1}, 2);
else
    toi_onset_samples = round(toi_onset * hosp_sr * 3600);
    toi_offset_samples = round(toi_offset * hosp_sr * 3600);
end

chan_natus_doi = data.trial{1,1}(1, toi_onset_samples:toi_offset_samples);
chan_natus_time_axis = 1/hosp_sr:1/hosp_sr:(size(chan_natus_doi,2))/hosp_sr;

% creating chan_br dataset of same length as chan_natus_doi
chan_br_time_axis = 1/br_sr:1/br_sr:(size(chan_br,2))/br_sr;

fig = figure('Visible', 'off'); plot(chan_natus_time_axis, zscore(chan_natus_doi), 'r'); hold on;
plot(chan_br_time_axis, zscore(chan_br), 'b');

title('native sampling  data');
legend('natus','BR');
saveas(fig, sprintf('%s_plot_native_sampling_firstpass.png', prefix));
close;

%% filtering Natus data
% chunk to data of interest
data2 = data;
data2.trial{1,1} = data2.trial{1, 1}(:, toi_onset_samples:toi_offset_samples);
data2.time{1,1} = data2.time{1, 1}(:, toi_onset_samples:toi_offset_samples);
data2.sampleinfo(1,2) = size(data2.time{1, 1}, 2);
data2.hdr.nSamples =  size(data2.time{1, 1}, 2);
data2.cfg.trl(1,2)  = size(data2.time{1, 1}, 2);

%filter possible line noise
cfg = [];
cfg.channel = channel_natus;
cfg.lpfilter = 'yes';
cfg.lpfreq = lpfreq;
cfg.lpfiltord = lpfiltord;
cfg.padding = padding;
cfg.bsfilter = 'yes';
cfg.bsfiltord = bsfiltord;
cfg.bsfreq = bsfreq;

data2 = ft_preprocessing(cfg, data2);

%% select channel number of interest from Natus data
chan_natus = data2.trial{1, 1}(1, :);

%% z-scoring both Natus and Blackrock
chan_natus = zscore(detrend((chan_natus)));
chan_br = zscore((chan_br));

%% Convert timing and size of the block from BR sampling to Natus
% sampling
edfsize = size(chan_natus, 2);
ns3size = size(chan_br, 2);
edf_block_size = round(ns3size / br_sr * hosp_sr);
seconds = ns3size/br_sr;
ns3time = 1/br_sr:1/br_sr:seconds;
edftime = 1/hosp_sr:1/hosp_sr:seconds;

edftime_full = 1/hosp_sr:1/hosp_sr:edfsize/hosp_sr;
targtime = 1/targ_sr:1/targ_sr:seconds;
targtime_edf = 1/targ_sr:1/targ_sr:edfsize/hosp_sr;
ds_block_size = size(targtime, 2);
     
%% Need to make adjustment if times don't line up (off by 1/hosp_sr)
if size(edftime, 2) ~= edf_block_size
    if size(edftime, 2) > edf_block_size
        edftime = edftime(1:edf_block_size);
    elseif size(edftime, 2) < edf_block_size
        edf_block_size = size(edftime, 2);
    end
end

%% Downsample Blackrock and Natus data
chan_natus_ds = interp1(edftime_full, chan_natus, targtime_edf, 'linear', 'extrap');
chan_br_ds = interp1(ns3time, chan_br, targtime, 'linear', 'extrap');

lastsamp = edfsize - edf_block_size;
chunk_starts = 1:1:lastsamp;
lastsamp_ds = size(chan_natus_ds, 2) - size(chan_br_ds, 2);
chunk_starts_ds = 1:1:lastsamp_ds;

%% plot downsampled data
% chunking natus data to time of interest for visual inspection
fig = figure('Visible', 'on'); plot(chan_natus_ds, 'r'); hold on;
plot(chan_br_ds, 'b'); hold off;
title('downsampled data');
legend('natus','BR');
saveas(fig,sprintf('%s_plot_downsampled_to_%d_Hz_firstpass.png', prefix, targ_sr));
close;

%% correlation analysis in down sampled data
cortab_x_ns_kern = [];

%Run the downsampled version, sampling every k samples
for k = kerns
    cortab_x_ds_kern = [];
    chunk_starts_ds_kern = 1:k:lastsamp_ds;
    
    disp(sprintf('Running: downsampled with kernel %d', k));
    % tStart_kern = tic;
    parfor chunk_ind = 1:size(chunk_starts_ds_kern, 2)
        chunk = chunk_starts_ds_kern(1, chunk_ind);
        % Interpolate edf to blackrock
        chunk_edf = chan_natus_ds(1, chunk:chunk+ds_block_size-1);
        % Loop through correlations at second intervals
        CX = corrcoef(chunk_edf, chan_br_ds);
        % See if anything correlates
        cortab_x_ds_kern(1, chunk_ind) = CX(2);
    end
    
    % If there isn't a clear point, try again with a smaller kernel
    if any(cortab_x_ds_kern > mean(cortab_x_ds_kern) + sd_thresh*std(cortab_x_ds_kern)) && k ~= 1
        disp('Kernel cross correlations complete')
        break
    else
        fprintf('No points above %d sd. Trying new kernel\n', sd_thresh)
    end
end

cortab_ds_kern_name = sprintf('%s_correlation_table_br_natus_alignment_downsampled_kernal_%d.mat', prefix, k);
% save correlation file from downsampled data
save(cortab_ds_kern_name, 'cortab_x_ds_kern');
kern_max_samp = chunk_starts_ds_kern(cortab_x_ds_kern > mean(cortab_x_ds_kern) + sd_thresh*std(cortab_x_ds_kern));

%% finding data from native space
chunks_native = [];
for ks = 1:size(kern_max_samp, 2)
    edf_kern_ind = find(edftime_full == targtime_edf(kern_max_samp(ks))); %only if 2^ data
    %temp = abs(edftime_full - targtime_edf(kern_max_samp(ks)));
    %edf_kern_ind = find(temp == min(temp)); 
    chunks_native = [chunks_native, edf_kern_ind-(window_ns*hosp_sr):edf_kern_ind+(window_ns*hosp_sr)];
end
chunks_native = chunks_native(chunks_native > 0);
chunks_native = chunks_native(chunks_native <= lastsamp);
chunks_native = unique(chunks_native);

%kern_max_samp_ns =  round((kern_max_samp/targ_sr)*hosp_sr);
%chunks_native = [kern_max_samp_ns(1)-500:kern_max_samp_ns(end)+500];

%% Running correlation in native space: centered around data of interest
% Now run the full version centered around the extracted timepoint

tic
parfor chunk_ind = 1:size(chunks_native, 2)
    chunk = chunks_native(1, chunk_ind);
    % Interpolate edf to blackrock
    chunk_edf = chan_natus(1, chunk:chunk+edf_block_size-1);
    chunk_interp = interp1(edftime, chunk_edf, ns3time, 'linear', 'extrap');
    % Loop through correlations at second intervals
    CX = corrcoef(chunk_interp, chan_br);
    % See if anything correlates
    cortab_x_ns(1, chunk_ind) = CX(2);
end
toc
ext_kern = chunks_native(cortab_x_ns == max(cortab_x_ns));
ext_kern2 = toi_onset_samples + ext_kern - 1;        
disp(' native sampling rate complete');

data_align = data;
data_align.trial{1, 1} = data.trial{1, 1}(:, ext_kern2:ext_kern2 + edf_block_size - 1);
data_align.time{1, 1} = 0:1/hosp_sr:(edf_block_size - 1)/hosp_sr;
data_align.sampleinfo(1,2) = size(data_align.time{1, 1}, 2);
data_align.hdr.nSamples =  size(data_align.time{1, 1}, 2);
data_align.cfg.trl(1,2)  = size(data_align.time{1, 1}, 2);

filename = strcat(prefix, '_data_align.mat');
save(filename, 'data_align', '-v7.3');

%% visual comparison of blackrock and clinical data
chan_natus_plot = interp1(edftime, data_align.trial{1, 1}(channel_natus, :), ns3time, 'linear', 'extrap');

gg = randi(size(ns3time, 2) - br_sr,[1 10]);
for i = gg
    fig =  figure('Visible', 'off');
    plot(detrend(chan_natus_plot(i:i+br_sr-1000))/max(abs(detrend(chan_natus_plot(i:i+br_sr-1)))), 'r'); hold on
    
    plot(detrend(chan_br(i:i+br_sr-1000))/max(abs(detrend(chan_br(i:i+br_sr-1)))), 'b');
    
    legend('natus', 'BR');
    saveas(fig,sprintf('%s_alignment_visual_%d.png', prefix, i));
    close
end

%% Add photodiode and audio trigger to aligned chunk of data
% input sampling rate you want the file to be saved with
targ_sr = 1024; 
% adjust sampling rate for photodiode
chan_PD = double(NS3.Data(PD_channel, :));
seconds = ns3size/br_sr;
ns3time = 1/br_sr:1/br_sr:seconds;
targtime = 1/targ_sr:1/targ_sr:seconds;

chan_PD_ds = interp1(ns3time, chan_PD, targtime, 'linear', 'extrap');

% resample Natus data
cfg.resamplefs = 1024; %frequency at which the data will be resampled (default = 256 Hz)
cfg.detrend    = 'no'; % do not detrend data, will be processed
cfg.demean     = 'no';
[data_align_1024] = ft_resampledata(cfg, data_align);

% select channel in natus that you like to overlap with 
chan_overlap_natus = 1;
data_align_1024.trial{1}(chan_overlap_natus,:) = chan_PD_ds;
plot(data_align_1024.trial{1}(chan_overlap_natus,:)) % checking if ok.

%% Save data into EDF
my_final_data = data_align_1024.trial{:};
%ft_write_data('SG1XX_ECoG_V1.edf', my_final_data, 'header', data_align.hdr);

header = data_align.hdr;
header.filetype = 'EDF+';
header.label = data.label;
ft_write_data('SG1XX_ECoG_V1.edf', my_final_data, 'header', header);

%% Save data into mat
filename = strcat(prefix, '_data_align_PD.mat');
save(filename, 'my_final_data', '-v7.3');

  