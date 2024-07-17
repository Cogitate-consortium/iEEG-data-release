function [data, triggers, labels, fs] = loadNlxData( pth )
% load nlx events/ncs and trim off dropouts
%   [data, triggers, labels] = loadNlxData( pth )
%       pth is full path to NLX data directory
%
%   Simon Henin, simon.henin@nyulangone.org
%   version 1: March 10th 2023


addpath('binaries');
if isempty(which('Nlx2MatCSC')),
    error('Could locate NLX binaries...');
end

if ~exist(pth, 'dir'),
    error('Could not locate the nlx directory at:\n\t %s', pth)
end

events_file = dir([pth '*.nev']);
if length(events_file) == 0,
    error('Could not locate events file (.nev) \nExiting...');
end
fprintf('loading events...');
events = getEvents( [pth events_file(1).name] );

idx_trg = [];
idx_trg(:,1) = 1:length(events);
% idx_trg(:,2) = [events.sample];
idx_trg(:,2) = [events(:, 1)];
idx_trg( find(diff(idx_trg(:,2)./1e6)<0.02)+1, :) = [];  % remove events within 20ms of initial burst
fprintf('done\n');

%%
ncs_files = dir([pth '*.ncs']);
if length(ncs_files) == 0,
    error('No ncs files.')
end

fprintf('getting timestamps....');
% load the first file for allocation
[timestamps,dataSamples] = getNCSData( [pth ncs_files(1).name], 0, 10e6, 2 );
fprintf('done\n');

fprintf('loading data....');
data = zeros( length(dataSamples), length(ncs_files));
labels = cell( length(ncs_files),1 );
num_files = length(ncs_files);
str_len = 1; 
for j=1:length(ncs_files),
    fprintf([repmat('\b',1, str_len) sprintf('%i/%i', j, num_files)]);
    str_len = length([num2str(j), num2str(num_files)])+1;

    [~, data(:, j)] = getNCSData( [pth ncs_files(j).name], 0, 10e6, 2 );
    labels{j} = regexprep( ncs_files(j).name, '.ncs', '');
end
fprintf('...done\n');


fprintf('adjusting timestamps/dropouts...');
% find the first dropout 
% idx = find( diff(timestamps) > 256000, 1);
idx = find( diff(timestamps) > 300000, 1);
if isempty(idx),
    idx = length(timestamps);
end
t = nan(512, idx);
t(1, :) = timestamps(1:idx);
t = t(:);
t = fillmissing(t, 'linear');
% check the sampling rate
data = data(1:length(t), :);


% find the last useable timestamp (e.g. > last t)
idx_end = find( double(idx_trg(:, 2)) > t(end), 1)-1;
if isempty(idx_end),
    idx_end = length(idx_trg);
end
fprintf('done\n');
fs = (length(t)+512) / ( (t(end)-t(1))/1e6);
fprintf('deduced sampling rate: %2.2f Hz\n', length(t) / ( (t(end)-t(1))/1e6));

fprintf('re-inserting trigger timestamps....');
% loop through events and use timestamp to mark triggers
triggers = zeros( length(t), 1);
str_len = 1;
for j=1:idx_end,
    fprintf([repmat('\b',1, str_len) sprintf('%i/%i', j, idx_end)]);
    str_len = length([num2str(j), num2str(idx_end)])+1;

    [~,idx] = min( abs(t-double(idx_trg(j, 2))) );
    if  abs((t(idx)-double(idx_trg(j, 2)))/1e6) > 0.001,
        warning('timestamp difference is larger than 1ms (=%2.2fms) at index: %i (inspect file)\n', abs((t(idx)-double(idx_trg(j, 2)))/1e6), j);
        fprintf('%s', repmat(' ', str_len, 1));
    end

    triggers(idx) = 1; 
end
fprintf('...done\n');
fprintf('conversion complete\n');

% NLX Utility functions
function events = getEvents( filename )
FieldSelection(1) = 1;%timestamps
FieldSelection(2) = 0;
FieldSelection(3) = 1;%ttls
FieldSelection(4) = 0;
FieldSelection(5) = 0;
ExtractHeader = 1;
ExtractMode = 1;
%ModeArray(1)=fromInd;
%ModeArray(2)=toInd;

if isunix,
    [timestamps,  ttls, header] = Nlx2MatEV_v3(filename, FieldSelection, ExtractHeader, ExtractMode);
else
    [timestamps,  ttls, header] = Nlx2MatEV(filename, FieldSelection, ExtractHeader, ExtractMode);
end
events=zeros(size(ttls,2),2);
events(:,1) = timestamps';
events(:,2) = ttls';


function [timestamps,dataSamples] = getNCSData(filename, fromInd, toInd, mode )

if nargin==3
    mode=2;
end

FieldSelection(1) = 1;%timestamps
FieldSelection(2) = 0;
FieldSelection(3) = 0;%sample freq
FieldSelection(4) = 0;
FieldSelection(5) = 1;%samples
ExtractHeader = 0;

ExtractMode = mode; % 2 = extract record index range; 4 = extract timestamps range.
ModeArray(1)=fromInd;
ModeArray(2)=toInd;

if isunix,
    [timestamps, dataSamples] = Nlx2MatCSC_v3(filename, FieldSelection, ExtractHeader, ExtractMode,ModeArray);
else
    % windows mex
    [timestamps, dataSamples] = Nlx2MatCSC(filename, FieldSelection, ExtractHeader, ExtractMode,ModeArray);
end

%flatten
dataSamples=dataSamples(:);
