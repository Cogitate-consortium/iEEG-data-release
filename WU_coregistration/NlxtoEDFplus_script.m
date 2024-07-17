%% Script to save NLX data to EDF+
% It first conversts the nlx data triggers and adjust for those not to be
% continuous. Then it saves data to EDF+ as needed to Cogitate project
% Urszula, March 2023

%% run conversion from NLX.
[data, triggers, labels, fs] = loadNlxData('\\');

%% Add trigger channel 
data_final = data';
data_final = [data_final; triggers'];
labels{111} = 'TRIG'; % input trigger channel
plot(data_final(111,:)); % check if looks as expected

%% Save data into EDF
% anonimize
data_final_hdr.hdr = ([]);

data_final_hdr.samplingrate = 2000; %fs
data_final_hdr.numchannels = 82;
data_final_hdr.numauxchannels = 0;
data_final_hdr.channels = char(labels);

data_final_hdr.year = 2022;
data_final_hdr.month = 5;
data_final_hdr.day = 5;
data_final_hdr.hour = 5;
data_final_hdr.minute = 5;
data_final_hdr.second = 5;
data_final_hdr.ID = 'X';
data_final_hdr.technician = 'X';
data_final_hdr.equipment = 'X';
data_final_hdr.subject = ([]);
data_final_hdr.ecg_ch = 82;
data_final_hdr.numdatachannels = 82;
data_final_hdr.numtimeframes = size(data_final, 2);
data_final_hdr.version = [];
data_final_hdr.millisecond = 0;

 
lab_write_edf('SG1XX_ECoG_V1_plus.edf',data_final,data_final_hdr)

  