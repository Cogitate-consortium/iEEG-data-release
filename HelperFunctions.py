"""
This script contains all helper functions for the preprocessing pipeline
Authors: Alex Lepauvre and Katarina Bendtz
alex.lepauvre@ae.mpg.de
katarina.bendtz@tch.harvard.edu
Dec 2020
"""

import os
from pathlib import Path
import numpy as np
import math
import matplotlib

import mne
from mne.viz import plot_alignment, snapshot_brain_montage
from mne.datasets import fetch_fsaverage
from mne_bids import convert_montage_to_mri, BIDSPath
from nibabel.freesurfer.io import read_geometry

import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.colors as mcolors

import pandas as pd

import json

# Set the MNE debug level:
mne.set_log_level(verbose='WARNING')


def detrend_runs(raw, njobs=1):
    """
    Detrend the run. If the raw data initially consisted of several files that were concatenated, 
    the detrending will be performed separately for each run. Otherwise, it will be done in one go.

    :param raw: mne raw object
    :param njobs: int, number of jobs to use for detrending
    :return: detrended mne raw object
    """
    # Check whether there are any annotations marking the merging:
    if len(np.where(raw.annotations.description == "BAD boundary")[0]) > 0:
        # Prepare a list to store the raws:
        raws_list = []
        # Extract the boundaries time stamps:
        boundaries_ts = raw.annotations.onset[np.where(raw.annotations.description == "BAD boundary")[0]]
        for i in range(boundaries_ts.shape[0] + 1):
            if i == 0:
                # Extract the segment:
                r = raw.copy().crop(0, boundaries_ts[i], include_tmax=False)
                # Apply detrending:
                r.apply_function(lambda ch: ch - np.mean(ch),
                                 n_jobs=njobs,
                                 channel_wise=True)
                # Append to the rest:
                raws_list.append(r)
            elif i == boundaries_ts.shape[0]:
                # Extract the segment:
                r = raw.copy().crop(boundaries_ts[i - 1], raw.times[-1], include_tmax=True)
                # Apply detrending:
                r.apply_function(lambda ch: ch - np.mean(ch),
                                 n_jobs=njobs,
                                 channel_wise=True)
                # Append to the rest:
                raws_list.append(r)
            else:
                # Extract the segment:
                r = raw.copy().crop(boundaries_ts[i - 1], boundaries_ts[i], include_tmax=False)
                # Apply detrending:
                r.apply_function(lambda ch: ch - np.mean(ch),
                                 n_jobs=njobs,
                                 channel_wise=True)
                # Append to the rest:
                raws_list.append(r)

        raw_detrend = mne.concatenate_raws(raws_list)
        # Make sure that no samples got missing:
        assert np.array_equal(raw_detrend.times, raw.times), "The times of the detrend data got altered!!!"
        assert all([ch == raw_detrend.ch_names[i] for i, ch in enumerate(raw.ch_names)]), \
            "The times of the detrend data got altered!!!"
    else:  # Otherwise, detrend the whole recording:
        raw_detrend = raw.copy()
        raw_detrend.apply_function(lambda ch: ch - np.mean(ch),
                                   n_jobs=njobs,
                                   channel_wise=True)

    return raw_detrend


def get_cmap_rgb_values(values, cmap=None, center=None):
    """
    Get RGB values for a list of values mapping onto a specified color bar. If a midpoint is set, 
    the color bar will be normalized accordingly.

    :param values: list of floats, list of values for which to obtain a color map
    :param cmap: string, name of the colormap
    :param center: float, value on which to center the colormap
    :return: list of rgb triplets, color for each passed value
    """
    if cmap is None:
        cmap = "RdYlBu_r"
    if center is None:
        center = np.mean([min(values), max(values)])
    # Create the normalization function:
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min(values), vcenter=center, vmax=max(values))
    colormap = colormaps.get_cmap(cmap)
    colors = [colormap(norm(value)) for value in values]

    return colors


def path_generator(directory):
    """
    Generate a folder if it doesn't exist.

    :param directory: string or pathlib path object, folder to create
    :return: directory path
    """
    #
    if not os.path.isdir(directory):
        # Creating the directory:
        os.makedirs(directory)

    return directory


def save_param(param, save_path, step, signal, file_prefix, file_extension):
    """
    Save the configuration parameters used to generate the data.

    :param param: dictionary, parameters used to generate the data
    :param save_path: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    """
    # Saving the config of this particular step:
    config_file_name = Path(save_path, '{}_desc-{}_ieeg{}.json'.format(file_prefix, step,
                                                                       file_extension.split('.')[0]))
    with open(str(config_file_name), 'w') as outfile:
        json.dump(param[step][signal], outfile)
    # Saving the entire config file as well:
    config_file_name = Path(save_path, '{}_desc-{}_ieeg{}.json'.format(file_prefix, 'all',
                                                                       file_extension.split('.')[0]))
    with open(str(config_file_name), 'w') as outfile:
        json.dump(param, outfile)

    return None


def mne_data_saver(data, param, save_root, step, signal, file_prefix, file_extension="-raw.fif"):
    """
    Save different instances of MNE objects.

    :param data: mne object (epochs, evoked, raw...), data to be saved
    :param param: dictionary, parameters used to generate the data
    :param save_root: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    :return: None
    """
    print("-" * 40)
    print("Saving mne object")

    # First, generating the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)
    # Generating the full file name:
    full_file_name = Path(save_path, '{}_desc-{}_ieeg{}'.format(file_prefix, step, file_extension))
    # Saving the data:
    data.save(full_file_name, overwrite=True)
    # Saving the config:
    save_param(param, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return None


def plot_channels_psd(raw, save_root, step, signal, file_prefix,
                      file_extension="-psd.png", plot_single_channels=False, channels_type=None):
    """
    Plot and save the power spectral density (PSD) of chosen electrodes.

    :param raw: mne raw object
    :param save_root: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    :param plot_single_channels: boolean, whether to plot single channels or only all of them superimposed
    :param channels_type: dict, list of the channels of interest
    :return: None
    """
    # Getting  the relevant channels:
    if channels_type is None:
        channels_type = {"ecog": True, "seeg": True}
    picks = mne.pick_types(raw.info, **channels_type)

    # Generate the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)
    # Create the name of the file:
    full_file_name = Path(save_path, '{}_desc-{}_ieeg{}'.format(file_prefix, step, file_extension))

    # ==========================================================
    # Plotting the psd from all the channels:
    raw.plot_psd(picks=picks, show=False)
    # Saving the figure:
    plt.savefig(full_file_name, dpi=300, transparent=True)
    plt.close()

    # ==========================================================
    # For all channels separately:
    if plot_single_channels:
        for pick in picks:
            raw.plot_psd(picks=[pick], show=False)
            full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}{}'.format(file_prefix, step,
                                                                           raw.ch_names[pick], file_extension))
            plt.savefig(full_file_name, dpi=300, transparent=True)
            plt.close()

    return None


def plot_bad_channels(raw, save_root, step, signal, file_prefix,
                      file_extension="bads.png", plot_single_channels=False, picks="bads"):
    """
    Plot the bad channels PSD and raw signal to show what is being discarded.

    :param raw: mne raw object
    :param save_root: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    :param plot_single_channels: boolean, whether to plot single channels or only all of them superimposed
    :param picks: list, list of the channels of interest
    :return: None
    """
    # Handling picks input:
    if picks == "bads":
        picks = raw.info["bads"]

    # Generate the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)
    # Create the name of the file:
    full_file_name = Path(save_path, '{}_desc-{}_ieeg{}'.format(file_prefix, step, file_extension))

    if len(picks) > 0:
        # Plotting the psd from all the channels:
        fig, axs = plt.subplots(1, 1)
        plt.suptitle("Bad channels: N= " + str(len(picks))
                     + " out of " + str(len(raw.info["ch_names"])))
        # Plotting the average of the good channels with standard error for reference. Downsampling, otherwise too many
        # data points:
        good_channels_data, times = \
            raw.copy().resample(100).get_data(picks=mne.pick_types(
                raw.info, ecog=True, seeg=True), return_times=True)
        mean_good_data = np.mean(good_channels_data.T, axis=1)
        ste_good_data = np.std(good_channels_data.T, axis=1) / \
                        np.sqrt(good_channels_data.T.shape[1])
        axs.plot(times, mean_good_data, alpha=0.5, color="black")
        axs.fill_between(times, mean_good_data - ste_good_data, mean_good_data + ste_good_data, alpha=0.2,
                         color="black")
        # Plotting the time series of all the bad channels:
        axs.plot(times, raw.copy().resample(100).get_data(
            picks=picks).T, alpha=0.8, label=picks)
        # Adding labels and legend:
        axs.set_xlabel("time (s)")
        axs.set_ylabel("amplitude")
        axs.legend()
        # Plotting the psd
        # raw.plot_psd(picks=picks, show=False, ax=axs[1:2])
        # Saving the figure:
        plt.savefig(full_file_name, transparent=True)
        plt.close()
        # For all channels separately:
        if plot_single_channels:
            # Looping through each channels:
            for pick in picks:
                # Plotting the psd from this channel:
                fig, axs = plt.subplots(2, 1)
                axs[0].plot(times, mean_good_data, alpha=0.5, color="black")
                axs[0].fill_between(times, mean_good_data - ste_good_data, mean_good_data + ste_good_data,
                                    alpha=0.2,
                                    color="black")
                axs[0].plot(times, raw.copy().resample(
                    200).get_data(picks=pick).T, label=pick)
                axs[0].set_xlabel("time (s)")
                axs[0].set_ylabel("amplitude")
                # Adding labels and legend:
                axs[0].set_xlabel("time (s)")
                axs[0].set_ylabel("amplitude")
                axs[0].legend()
                # Plotting the psd
                raw.plot_psd(picks=pick, show=False, ax=axs[1])
                full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}{}'.format(file_prefix, step,
                                                                               raw.ch_names[pick], file_extension))
                plt.savefig(full_file_name, transparent=True)
                plt.close()

    return None


def custom_car(raw, reference_channel_types=None, target_channel_types=None):
    """
    Compute a custom common average reference (CAR) by averaging the amplitude across specific channel types and 
    subtracting it from all channels at each time point.

    :param raw: mne raw object, contains the data to be rereferenced
    :param reference_channel_types: dict, specifying the channels types to take as reference
    :param target_channel_types: dict, specifying the channels types to apply reference to
    :return: mne raw object, modified instance of the mne object
    """
    # Handling empty input:
    if reference_channel_types is None:
        reference_channel_types = {'ecog': True}
    if target_channel_types is None:
        target_channel_types = {'ecog': True}

    # Setting the electrodes to apply from and to
    ref_from = mne.pick_types(raw.info, **reference_channel_types)
    ref_to = mne.pick_types(raw.info, **target_channel_types)

    # Fetching the data. Using _data instead of get_data, because that enables modifying data in place:
    data = raw._data

    # Compute and apply ref:
    ref_data = data[..., ref_from, :].mean(-2, keepdims=True)
    data[..., ref_to, :] -= ref_data

    # Logging the custom ref:
    raw.info['custom_ref_applied'] = 1

    return raw


def notch_filtering(raw, njobs=1, frequency=60, remove_harmonics=True, filter_type="fir",
                    cutoff_lowpass_bw=None, cutoff_highpass_bw=None, channel_types=None):
    """
    Filter the raw data according to the set parameters.

    :param raw: mne raw object, continuous data
    :param njobs: int, number of jobs to preprocess the filtering in parallel threads
    :param frequency: int or float, frequency to notch out
    :param remove_harmonics: boolean, whether or not to remove all the harmonics of the declared frequency
    :param filter_type: string, type of filter to use, "iir" or "fir"
    :param cutoff_lowpass_bw: float, frequency for low pass (only used if type is iir)
    :param cutoff_highpass_bw: float, frequency for high pass (only used if type is iir)
    :param channel_types: dict, type of channels to notch filter, boolean for the channel types
    :return: mne raw object
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Filtering the data:
    # Getting the harmonics frequencies if any:
    if channel_types is None:
        channel_types = {"ecog": True, "seeg": True, "exclude": "bads"}
    if remove_harmonics:
        freq_incl_harmonics = [
            frequency * i for i in range(1, int((raw.info['sfreq'] / 2) // frequency) + 1)]
        frequency = freq_incl_harmonics

    # Selecting the channels to filter:
    picks = mne.pick_types(raw.info, **channel_types)
    if filter_type.lower() == "fir":
        # Applying FIR if FIR in parameters
        raw.notch_filter(frequency, filter_length='auto',
                         phase='zero', n_jobs=njobs, picks=picks)  # Default method is FIR
    elif filter_type.lower() == "butterworth_4o":  # Applying butterworth 4th order
        # For the iir methods, mne notch_filter does not support to pass several frequencies at the same time.
        # It also does not support having customized cutoff frequencies.
        # We therefore
        # 1. call the filter function to perform a custom butterworth 4th order
        # (default if method is set to "iir" and no iir_params parameter is given)
        # bandpass filter (notch_cutoff_hp Hz - notch_cutoff_lp Hz)
        if cutoff_lowpass_bw != 0:
            raw.filter(cutoff_lowpass_bw, cutoff_highpass_bw,
                       phase='zero', method='iir', n_jobs=njobs, picks=picks)
        else:
            raw.notch_filter(frequency, method='iir', n_jobs=njobs)
        # If there are harmonics to filter out as well:
        if remove_harmonics:
            # Now drop the first frequency from the frequencies:
            frequencies = frequency[1:]
            # 2. call the notch_filter function for each of the harmonics to perform the filtering of the harmonics.
            # Note that we here use the standard bandwidth freq/200.
            for freq in frequencies:
                raw.notch_filter(freq, method='iir', n_jobs=njobs, picks=picks)

    return raw


def create_metadata_from_events(epochs, metadata_column_names):
    """
    Parse the events found in the epochs descriptions to create the metadata. The columns of the metadata are generated 
    based on the metadata column names.

    :param epochs: mne epochs object, epochs for which the metadata will be generated
    :param metadata_column_names: list of strings, name of the columns of the metadata
    :return: mne epochs object with added metadata
    """

    # Getting the event description of each single trial
    trials_descriptions = [[key for key in epochs.event_id.keys() if epochs.event_id[key] == event]
                           for event in epochs.events[:, 2]]
    trial_descriptions_parsed = [description[0].split(
        "/") for description in trials_descriptions]
    # Making sure that the dimensions of the trials description is consistent across all trials:
    if len(set([len(vals) for vals in trial_descriptions_parsed])) > 1:
        raise ValueError('dimension mismatch in event description!\nThe forward slash separated list found in the '
                         'epochs description has inconsistent length when parsed. Having different number of '
                         'descriptors for different trials is not yet supported. Please make sure that your events '
                         'description are set accordingly')
    if len(metadata_column_names) != len(trial_descriptions_parsed[0]):
        raise ValueError("The number of meta data columns you have passed doesn't match the number of descriptors for\n"
                         "each trials. Make sure you have matching numbers. In doubt, go and check the events file in\n"
                         "the BIDS directory")
    if len(trial_descriptions_parsed) != len(epochs):
        raise ValueError("Somehow, the number of trials descriptions found in the epochs object doesn't match the "
                         "number of trials in the same epochs. I have no idea how you managed that one champion, so I "
                         "can't really help here")

    # Convert the trials description to a pandas dataframe:
    epochs.metadata = pd.DataFrame.from_records(
        trial_descriptions_parsed, columns=metadata_column_names)

    return epochs


def epoching(raw, events, events_dict, picks="all", tmin=-0.5, tmax=2.0, events_not_to_epoch=None,
             baseline=(None, 0.0), reject_by_annotation=True, meta_data_column=None):
    """
    Perform epoching according to the provided parameters.

    :param raw: mne raw object, data to epoch
    :param events: mne events numpy array, three columns: event time stamp, event ID
    :param events_dict: dict, mapping between the events ID and their descriptions
    :param picks: string or list of int or list of strings, channels to epoch
    :param tmin: float, how long before each event of interest to epoch (in seconds)
    :param tmax: float, how long after each event of interest to epoch (in seconds)
    :param events_not_to_epoch: list of strings, name of the events not to epoch
    :param baseline: tuple of floats, passed to the mne epoching function to apply baseline correction
    :param reject_by_annotation: boolean, whether or not to discard trials based on annotations
    :param meta_data_column: list of strings, list of the column names of the metadata
    :return: mne epochs object
    """
    if picks == "all":
        picks = raw.info["ch_names"]
    if isinstance(baseline, list):
        baseline = tuple(baseline)
    print('Performing epoching')

    # We only want to epochs certain events. The config file specifies which ones should not be epoched (fixation
    # for ex). The function below returns only the events we are interested in!
    if events_not_to_epoch is not None:
        events_of_interest = {key: events_dict[key] for key in events_dict if not
        any(substring in key for substring in events_not_to_epoch)}
    else:
        events_of_interest = events_dict

    # Epoching the data:
    # The events are all events and the event_id are the events which we want to use for the
    # epoching. Since we are passing a dictionary we can also use the provided keys to acces
    # the events later
    epochs = mne.Epochs(raw, events=events, event_id=events_of_interest, tmin=tmin,
                        tmax=tmax, baseline=baseline, picks=picks,
                        reject_by_annotation=reject_by_annotation)
    # Dropping the bad epochs if there were any:
    epochs.drop_bad()
    # Adding the meta data to the table. The meta data are created by parsing the events strings, as each substring
    # contains specific info about the trial:
    if meta_data_column is not None:
        epochs = create_metadata_from_events(epochs, meta_data_column)

    return epochs


def compute_hg(raw, frequency_range=None, njobs=1, bands_width=10, channel_types=None,
               do_baseline_normalization=True):
    """
    This function computes the high gamma signal by filtering the signal in bins between a defined frequency range.
    In each frequency bin, the hilbert transform is applied and the amplitude is extracted (i.e. envelope). The envelope
    is then normalized by dividing by the mean amplitude over the entire time series (within each channel separately).
    The amplitude is then averaged across frequency bins. This follows the approach described here:
    https://www.nature.com/articles/s41467-019-12623-6#Sec10

    :param raw: mne raw object, raw signal
    :param frequency_range: list of list of floats, frequency of interest
    :param bands_width: float or int, width of the frequency bands to loop over
    :param njobs: int, number of parallel processes to compute the high gammas
    :param channel_types: dict, name of the channels for which the high gamma should be computed
    :param do_baseline_normalization: bool, whether to do baseline normalization
    :return: mne raw object, dictionary containing raw objects with high gamma in the different frequency bands
    """

    def divide_by_average(data):
        if not isinstance(data, np.ndarray):
            raise TypeError('Input value must be an ndarray')
        if data.ndim != 2:
            raise TypeError('The data should have two dimensions!')
        else:
            norm_data = data / data.mean(axis=1)[:, None]
            # Check that the normalized data are indeed what they should be. We are dividing the data by a constant.
            # So if we divide the normalized data by the original data, we should get a constant:
            if len(np.unique((data[0, :] / norm_data[0, :]).round(decimals=13))) != 1:
                raise Exception(
                    "The normalization of the frequency band went wrong! The division of normalized vs non "
                    "normalized data returned several numbers, which shouldn't be the case!")
            return norm_data

    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    if frequency_range is None:
        frequency_range = [70, 150]
    print('-' * 40)
    print('Welcome to frequency bands computation')
    print(njobs)

    # ==========================================================
    # Handle channels
    # Selecting the channels for which to compute the frequency band:
    picks = mne.pick_types(raw.info, **channel_types)
    # Getting the index of the channels for which the frequency band should NOT be computed
    not_picks = [ind for ind, ch in enumerate(
        raw.info["ch_names"]) if ind not in picks]
    # Creating copies of the raw for the channels for which the frequency band should be computed, and another for
    # the channels for which it shouldn't be computed, to avoid messing up the channels indices:
    freq_band_raw = raw.copy().pick(picks)
    if len(not_picks) != 0:
        rest_raw = raw.copy().pick(not_picks)

    # ==========================================================
    # Create frequency bins:
    # We first create the dictionary to store the final data:
    bins_amp = []
    # We then create the frequency bins to loop over:
    bins = []
    for i, freq in enumerate(range(frequency_range[0], frequency_range[1], bands_width)):
        bins.append([freq, freq + bands_width])

    # ==========================================================
    # Compute HG for each frequency bin:
    for freq_bin in bins:
        print('-' * 40)
        print('Computing the frequency in band: ' + str(freq_bin))
        # Filtering the signal and apply the hilbert:
        print('Computing the envelope amplitude')
        bin_power = freq_band_raw.copy().filter(freq_bin[0], freq_bin[1],
                                                n_jobs=njobs).apply_hilbert(envelope=True)

        # Now, dividing the amplitude by the mean, channel wise:
        if do_baseline_normalization:
            bins_amp.append(divide_by_average(bin_power[:][0]))
        else:
            bins_amp.append(bin_power[:][0])

    # Deleting unused variables to make some space:
    del bin_power
    info = freq_band_raw.info

    # ==========================================================
    # Average across bins:
    # Converting this to 3D numpy array:
    bins_amp = np.array(bins_amp)
    # Finally, all the bands must be averaged back together:
    frequency_band = bins_amp.mean(axis=0)

    # ==========================================================
    # Testing of the output:
    # Checking that the averaging across frequency bands works as we want it and if not raise an error:
    for i in range(0, 100):
        ch_ind = np.random.randint(0, bins_amp.shape[1])
        sample_ind = np.random.randint(0, bins_amp.shape[2])
        # Extracting the data of all each freq band and compute the mean thereof:
        test_avg = np.mean(bins_amp[:, ch_ind, sample_ind])
        observer_avg = frequency_band[ch_ind, sample_ind]
        if (test_avg - observer_avg).round(decimals=14) != 0:
            print(test_avg - observer_avg)
            raise Exception("There was an issue in the averaging across frequency bands in the frequency "
                            "bands computations!")

    # Recreating mne raw object:
    hg_signal = mne.io.RawArray(frequency_band, info)
    # Adding back the untouched channels:
    if len(not_picks) != 0:
        hg_signal.add_channels([rest_raw])

    return hg_signal


def compute_erp(raw, frequency_range=None, njobs=1, channel_types=None, **kwargs):
    """
    Compute the event-related potential (ERP) by low-pass filtering the raw signal.

    :param raw: mne raw object, raw signal
    :param frequency_range: list of list of floats, frequency of interest
    :param njobs: int, number of parallel processes to compute the ERP
    :param channel_types: dict, type of channel for which the ERP should be computed
    :param kwargs: arguments that can be passed to the mne raw.filter function
    :return: mne raw object, computed ERP signal
    """
    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    if frequency_range is None:
        frequency_range = [0, 30]
    print('-' * 40)
    print('Welcome to erp computation')
    print(njobs)
    picks = mne.pick_types(raw.info, **channel_types)
    # Filtering the signal according to the passed parameters:
    erp_raw = raw.copy().filter(
        frequency_range[0], frequency_range[1], n_jobs=njobs, picks=picks, **kwargs)

    return erp_raw


def add_fiducials(montage, fs_directory, subject_id):
    """
    Add the estimated fiducials to the montage and compute the transformation.

    :param montage: mne raw object, data to which the fiducials should be added
    :param fs_directory: path string, path to the freesurfer directory
    :param subject_id: string, name of the subject
    :return: mne raw object and transformation
    """
    # If the coordinates are in mni_tal coordinates:
    if montage.get_positions()['coord_frame'] == "mni_tal":
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = Path(sample_path, 'subjects')
        montage.add_mni_fiducials(subjects_dir)
        trans = 'fsaverage'
    elif montage.get_positions()['coord_frame'] == 'ras':
        # we need to go from scanner RAS back to surface RAS (requires recon-all)
        convert_montage_to_mri(montage, subject_id, subjects_dir=fs_directory)
        # Add the estimated fiducials:
        montage.add_estimated_fiducials(subject_id, fs_directory)
        trans = mne.channels.compute_native_head_t(montage)
    else:
        # If the montage space is unknown, assume that it is in MRI coordinate frame:
        montage.add_estimated_fiducials(subject_id, fs_directory)
        trans = mne.channels.compute_native_head_t(montage)

    return montage, trans


def plot_electrode_localization(mne_object, subject, fs_dir, param, save_root, step, signal, file_prefix,
                                file_extension='-loc.png', channels_to_plot=None,
                                plot_elec_name=False):
    """
    Plot and save the electrode localization.

    :param mne_object: mne object (raw, epochs, evoked...), contains the mne object with the channels info
    :param subject: string, subject ID
    :param fs_dir: string or path object, freesurfer directory containing the subject's data
    :param param: dict, contains analysis parameters
    :param save_root: string, directory to save the figures
    :param step: string, name of the analysis step for saving the parameters
    :param signal: string, name of the signal
    :param file_prefix: string, prefix for file saving
    :param file_extension: string, ending of the file name
    :param channels_to_plot: list, contains the different channels to plot
    :param plot_elec_name: string, whether or not to print the electrodes names onto the snapshot
    :return: None
    """
    if channels_to_plot is None:
        channels_to_plot = ["ecog", "seeg"]
    if mne_object.get_montage().get_positions()['coord_frame'] == "mni_tal":
        subject = "fsaverage"
        fetch_fsaverage(subjects_dir=fs_dir, verbose=True)
    # First, generating the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)

    # Adding the estimated fiducials and compute the transformation to head:
    montage, trans = add_fiducials(mne_object.get_montage(), fs_dir, subject)
    mne_object.set_montage(montage, on_missing="warn")

    brain_snapshot_files = []
    # Setting the two views
    snapshot_orientations = {
        "left": {"azimuth": 180, "elevation": None},
        "front": {"azimuth": 90, "elevation": None},
        "right": {"azimuth": 0, "elevation": None},
        "back": {"azimuth": -90, "elevation": None},
        "top": {"azimuth": 0, "elevation": 180},
        "bottom": {"azimuth": 0, "elevation": -180}
    }
    # We want to plot the seeg and ecog channels separately:
    for ch_type in channels_to_plot:
        try:
            data_to_plot = mne_object.copy().pick(ch_type)
        except ValueError:
            print("No {0} channels for this subject".format(ch_type))
            continue
        # Plotting the brain surface with the electrodes and making a snapshot
        if ch_type == "ecog":
            fig = plot_alignment(data_to_plot.info, subject=subject, subjects_dir=fs_dir,
                                 surfaces=['pial'], coord_frame='head', trans=trans)
        else:
            fig = plot_alignment(data_to_plot.info, subject=subject, subjects_dir=fs_dir,
                                 surfaces=['white'], coord_frame='head', trans=trans)

        for ori in snapshot_orientations.keys():
            if plot_elec_name:
                # Generating the full file name:
                full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}_view-{}_names{}'.format(file_prefix,
                                                                                             step,
                                                                                             ori, ch_type,
                                                                                             file_extension))
            else:
                full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}_view-{}{}'.format(file_prefix, step, ori, ch_type,
                                                                                       file_extension))
            mne.viz.set_3d_view(fig, **snapshot_orientations[ori])
            xy, im = snapshot_brain_montage(
                fig, data_to_plot.info, hide_sensors=False)
            fig_2, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(im)
            ax.set_axis_off()
            if plot_elec_name is True:
                for ch in list(xy.keys()):
                    ax.annotate(ch, xy[ch], fontsize=14, color="white",
                                xytext=(-5, -5), xycoords='data', textcoords='offset points')
            plt.savefig(full_file_name, transparent=True)
            plt.close()
            brain_snapshot_files.append(full_file_name)
        mne.viz.close_3d_figure(fig)
    # Saving the config:
    save_param(param, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return None


def roi_mapping(mne_object, list_parcellations, subject, fs_dir, param, save_root, step, signal, file_prefix,
                file_extension='mapping.csv'):
    """
    Map the electrodes on different atlases.

    :param mne_object: mne raw or epochs object, object containing the montage to extract the roi
    :param list_parcellations: list of string, list of the parcellation files to use for the mapping
    :param subject: string, name of the subject to access the fs recon
    :param fs_dir: string or path object, path to the freesurfer directory containing all the participants
    :param param: dictionary, parameters used to generate the data
    :param save_root: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    :return: dict of dataframes, one dataframe per parcellation with the mapping between roi and channels
    """

    # Generate the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)

    labels_df = {parcellation: pd.DataFrame()
                 for parcellation in list_parcellations}
    for parcellation in list_parcellations:
        if mne_object.get_montage().get_positions()['coord_frame'] == "mni_tal":
            sample_path = mne.datasets.sample.data_path()
            subjects_dir = Path(sample_path, 'subjects')
            # Convert the montge from mni to mri:
            montage = mne_object.get_montage()
            montage.apply_trans(mne.transforms.Transform(fro='mni_tal', to='mri', trans=np.eye(4)))
            labels, _ = \
                mne.get_montage_volume_labels(montage, "fsaverage", subjects_dir=subjects_dir,
                                              aseg=parcellation)
        elif mne_object.get_montage().get_positions()['coord_frame'] == 'ras':
            montage = mne_object.get_montage()
            # we need to go from scanner RAS back to surface RAS (requires recon-all)
            convert_montage_to_mri(montage, subject, subjects_dir=fs_dir)
            labels, _ = mne.get_montage_volume_labels(montage, subject, subjects_dir=fs_dir, aseg=parcellation)
        else:
            labels, _ = mne.get_montage_volume_labels(
                mne_object.get_montage(), subject, subjects_dir=fs_dir, aseg=parcellation)
        # Appending the electrodes roi to the table:
        labels_df[parcellation] = (
            pd.concat([pd.DataFrame({
                "channel": ch,
                "region": "/".join(labels[ch])
            }, index=[ind])
                for ind, ch in enumerate(labels.keys())]))
    # Creating the directory if it doesn't exists:
    if not os.path.isdir(save_path):
        # Creating the directory:
        os.makedirs(save_path)
    # Looping through the different mapping:
    for mapping in labels_df.keys():
        # Create the name of the file:
        full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}{}'.format(file_prefix, step, mapping,
                                                                       file_extension))
        # Saving the corresponding mapping:
        labels_df[mapping].to_csv(Path(full_file_name), index=False)
    # Saving the config:
    save_param(param, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return labels_df


def description_ch_rejection(raw, bids_path, channels_description, discard_bads=True):
    """
    Discard channels based on the descriptions found in the _channel.tsv file in the BIDS dataset.

    :param raw: mne raw object, contains the data and channels to investigate
    :param bids_path: mne_bids BIDSPath object, path to the _channel.tsv file
    :param channels_description: str or list, channels descriptions to set as bad channels
    :param discard_bads: boolean, whether or not to discard the channels that were marked as bad
    :return: tuple (mne raw object, list of bad channels)
    """
    if isinstance(channels_description, str):
        channels_description = [channels_description]

    # Loading the channels description file:
    channel_info_file = Path(bids_path.directory,
                             'sub-{}_ses-{}_task-{}_channels.tsv'.format(bids_path.subject, bids_path.session,
                                                                         bids_path.task))
    channel_info = pd.read_csv(channel_info_file, sep="\t")
    # Looping through the passed descriptions:
    desc_bad_channels = []
    for desc in channels_description:
        # Looping through each channel:
        for ind, row in channel_info.iterrows():
            if isinstance(row["status_description"], str):
                if desc in row["status_description"]:
                    desc_bad_channels.append(row["name"])

    # Discarding the channels that were marked as bad as well
    if discard_bads:
        for ind, row in channel_info.iterrows():
            if row["status"] == "bad":
                desc_bad_channels.append(row["name"])
    # Remove any redundancies:
    bad_channels = list(set(desc_bad_channels))
    # Set these channels as bad:
    raw.info['bads'].extend(bad_channels)

    return raw, bad_channels


def laplace_mapping_validator(mapping, data_channels):
    """
    Check the mapping against the channels found in the data. Raise an error if there are inconsistencies.

    :param mapping: dict, contains the mapping between channels to reference and which channels to use for the reference
    :param data_channels: list of string, list of the channels found in the data
    :return: None
    """
    if not all([channel in data_channels for channel in mapping.keys()]) \
            or not all([mapping[channel]["ref_1"] in data_channels or mapping[channel]["ref_1"] is None
                        for channel in mapping.keys()]) \
            or not all([mapping[channel]["ref_2"] in data_channels or mapping[channel]["ref_2"] is None
                        for channel in mapping.keys()]):
        # Printing the name of the channels that are not present in the data:
        print("The following channels are present in the mapping but not in the data:")
        print([channel for channel in mapping.keys() if channel not in data_channels])
        print([mapping[channel]["ref_1"]
               for channel in mapping.keys() if mapping[channel]["ref_1"] not in data_channels])
        print([mapping[channel]["ref_2"]
               for channel in mapping.keys() if mapping[channel]["ref_2"] not in data_channels])
        raise Exception("The mapping contains channels that are not in the data!")
    # Checking that there is never the case of having both reference as None:
    if any([mapping[channel]["ref_1"] is None and mapping[channel]["ref_2"] is None for channel in mapping.keys()]):
        invalid_channels = [channel for channel in mapping.keys() if
                            mapping[channel]["ref_1"] is None and mapping[channel]["ref_2"] is None]
        mne.utils.warn("The channels {0} have two None reference. They will be set to bad! If this is not intended,"
                       "please review your mapping!".format(invalid_channels))

    return None


def remove_bad_references(reference_mapping, bad_channels, all_channels):
    """
    Integrate bad channels information to the reference mapping. Update the mapping such that channels with bad 
    references are excluded.

    :param reference_mapping: dict, contain for each channel the reference channel according to the scheme
    :param bad_channels: string list, list of the channels that are marked as bad
    :param all_channels: string list, list of all the channels
    :return: tuple (new_reference_mapping, new_bad_channels)
    """
    new_reference_mapping = reference_mapping.copy()
    # Looping through each channel to reference to combine bad channels information:
    for channel in reference_mapping.keys():
        # If the channel being referenced is bad, then it is bad:
        if channel in bad_channels:
            print("The channel {} to reference is bad and will be ignored".format(channel))
            new_reference_mapping.pop(channel)
            continue
        # If both the references are bad, setting the channel to bad too, as it can't be referenced!
        elif reference_mapping[channel]["ref_1"] in bad_channels \
                and reference_mapping[channel]["ref_2"] in bad_channels:
            print("The reference channels {} and {} to reference {} are both bad and {} cannot be referenced".format(
                reference_mapping[channel]["ref_1"], reference_mapping[channel]["ref_2"], channel, channel))
            new_reference_mapping.pop(channel)
            continue
        # But if only one of the two reference is bad, setting that one ref to None:
        elif reference_mapping[channel]["ref_1"] in bad_channels:
            new_reference_mapping[channel]["ref_1"] = None
        elif reference_mapping[channel]["ref_2"] in bad_channels:
            new_reference_mapping[channel]["ref_2"] = None

        # As a result of setting one of the reference to None if bad, some channels located close to edges might have
        # only None as references, in which case they can't be referenced. This channels need to be removed from the
        # mapping
        if new_reference_mapping[channel]["ref_1"] is None and new_reference_mapping[channel]["ref_2"] is None:
            print("The reference channels {} cannot be referenced, because both surrounding channels are None as a "
                  "result of bad channels removal".format(channel))
            new_reference_mapping.pop(channel)

    # Removing any duplicates there might be (some channels might be both bad themselves and surrounded by bads):
    new_bad_channels = [channel for channel in all_channels if channel not in list(new_reference_mapping.keys())]
    # Compute the proportion of channels that are bad because surrounded by bad:
    laplace_bad_cnt = len(new_bad_channels) - len(bad_channels)
    # Print some info about what was discarded:
    print("{0} dropped because bad or surrounded by bad".format(new_bad_channels))
    print("{} bad channels".format(len(bad_channels)))
    print("{} surrounded by bad channels".format(laplace_bad_cnt))

    return new_reference_mapping, new_bad_channels


def laplace_ref_fun(to_ref, ref_1=None, ref_2=None):
    """
    This function computes the laplace reference by subtracting the mean of ref_1 and ref_2 to the channel to reference:
    ch = ch - mean(ref_1, ref_2)
    The to-ref channel must be a matrix with dim: [channel, time]. The ref_1 and ref_2 must have the same dimensions AND
    the channel rows must match the order. So if you want to reference G2 with G1 and G2 and well as G3 with G2 and G4,
    then your matrices must be like so:
    to_ref:                ref_1              ref_2
    [                 [                       [
    G2 ..........     G1 ..............       G3 ..............
    G3 ..........     G2 ..............       G4 ..............
    ]                 ]                       ]
    :param to_ref: numpy array, contains the data to reference
    :param ref_1: numpy array, contains the first ref to do the reference
    :param ref_2: numpy array, contains the second ref to do the reference
    :return: numpy array, referenced data
    """
    # Check that the sizes match:
    if not to_ref.shape == ref_1.shape == ref_2.shape:
        raise Exception("The dimension of the data to subtract do not match the data!")
    referenced_data = to_ref - np.nanmean([ref_1, ref_2], axis=0)

    return referenced_data


def project_elec_to_surf(raw, subjects_dir, subject):
    """
    Project surface electrodes onto the brain surface to avoid having them floating.

    :param raw: mne raw object
    :param subjects_dir: path or string, path to the freesurfer subject directory
    :param subject: string, name of the subject
    :return: mne raw object
    """
    # Loading the left and right pial surfaces:
    if raw.get_montage().get_positions()['coord_frame'] == "mri":
        # Create the file names to the directory:
        lhfile = Path(subjects_dir, subject, "surf", "lh.pial")
        rhfile = Path(subjects_dir, subject, "surf", "rh.pial")
        left_surf = read_geometry(str(lhfile))
        right_surf = read_geometry(str(rhfile))
    elif raw.get_montage().get_positions()['coord_frame'] == "mni_tal":
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = Path(sample_path, 'subjects')
        fetch_fsaverage(subjects_dir=str(subjects_dir), verbose=True)  # Downloading the data if needed
        subject = "fsaverage"
        lhfile = Path(subjects_dir, subject, "surf", "lh.pial")
        rhfile = Path(subjects_dir, subject, "surf", "rh.pial")
        left_surf = read_geometry(str(lhfile))
        right_surf = read_geometry(str(rhfile))
    else:
        raise Exception("You have passed a montage space that is not supported! Either MNI or T1!")

    # Getting the surface electrodes:
    ecog_picks = mne.pick_types(raw.info, ecog=True)
    ecog_channels = [raw.ch_names[pick] for pick in ecog_picks]
    # Get the montage:
    montage = raw.get_montage()
    # Looping through each of these channels:
    for channel in ecog_channels:
        # Get the channel index
        ch_ind = montage.ch_names.index(channel)
        # Get the channel coordinates:
        ch_coord = montage.dig[ch_ind]["r"]
        # Checking if the coordinates are NAN:
        if math.isnan(ch_coord[0]):
            continue
        if ch_coord[0] < 0:
            # Compute x y and z distances to each vertex in the surface:
            b_x = np.absolute(left_surf[0][:, 0] - ch_coord[0] * 1000)
            b_y = np.absolute(left_surf[0][:, 1] - ch_coord[1] * 1000)
            b_z = np.absolute(left_surf[0][:, 2] - ch_coord[2] * 1000)
            # Find the shortest distance:
            d = np.sqrt(np.sum([np.square(b_x), np.square(b_y), np.square(b_z)], axis=0))
            # Get the index of the smallest one:
            min_vert_ind = np.argmin(d)
            montage.dig[ch_ind]["r"] = np.squeeze(np.array(
                [left_surf[0][min_vert_ind, 0] * 0.001, left_surf[0][min_vert_ind, 1] * 0.001,
                 left_surf[0][min_vert_ind, 2] * 0.001]))
        else:
            # Compute x y and z distances to each vertex in the surface:
            b_x = np.absolute(right_surf[0][:, 0] - ch_coord[0] * 1000)
            b_y = np.absolute(right_surf[0][:, 1] - ch_coord[1] * 1000)
            b_z = np.absolute(right_surf[0][:, 2] - ch_coord[2] * 1000)
            # Find the shortest distance:
            d = np.sqrt(np.sum([np.square(b_x), np.square(b_y), np.square(b_z)], axis=0))
            # Get the index of the smallest one:
            min_vert_ind = np.argmin(d)
            montage.dig[ch_ind]["r"] = np.squeeze(np.array(
                [right_surf[0][min_vert_ind, 0] * 0.001, right_surf[0][min_vert_ind, 1] * 0.001,
                 right_surf[0][min_vert_ind, 2] * 0.001]))
    # Adding the montage back to the raw object:
    raw.set_montage(montage, on_missing="warn")

    return raw


def laplacian_referencing(raw, reference_mapping, channel_types=None,
                          n_jobs=1, relocate_edges=True,
                          subjects_dir=None, subject=None):
    """
    Performs laplacian referencing by subtracting the average of two neighboring electrodes to the
    central one. So for example, if you have electrodes G1, G2, G3, you can reference G2 as G2 = G2 - mean(G1, G2).
    The user can pass a mapping in the format of a dictionary. The dictionary must have the structure:
    {
    "ch_to_reference": {
        "ref_1": "ref_1_ch_name" or None,
        "ref_1": "ref_1_ch_name" or None,
        },
    ...
    }
    If the user doesn't pass a mapping, he will be given the opportunity to generate it manually through command line
    input. If no mapping exists, we recommend using the command line to generate it, as the formating is then readily
    consistent with the needs of the function. The function so far only works with raw object, not epochs nor evoked

    :param raw: mne raw object, contains the data to reference
    :param reference_mapping: dict or None, dict of the format described above or None
    :param channel_types: dict, which channels to consider for the referencing
    :param n_jobs: int, number of jobs to compute the mapping
    :param relocate_edges: boolean, whether or not to relocate the electrodes that have only one ref
    :param subjects_dir: string, directory to the freesurfer data
    :param subject: string, name of the subject to access the right surface
    :return: tuple (mne raw object, reference_mapping, bad_channels)
    """
    # Get the channels of interest.
    if channel_types is None:
        channel_types = {"ecog": True, "seeg": True, "exclude": []}
    channels_of_int = [raw.ch_names[ind] for ind in mne.pick_types(raw.info, **channel_types)]

    # Validate the reference mapping:
    laplace_mapping_validator(reference_mapping, channels_of_int)
    # Adjust reference mapping based on bad channels information:
    reference_mapping, bad_channels = remove_bad_references(reference_mapping, raw.info["bads"], channels_of_int)

    # ------------------------------------------------------------------------------------------------------------------
    # Performing the laplace reference:
    # Extract data to get the references and avoid issue with changing in place when looping:
    ref_data = raw.get_data()
    data_chs = raw.ch_names
    # Get the size of a channel matrix to handle the absence of ref_2 for corners:
    mat_shape = np.squeeze(raw.get_data(picks=0)).shape
    empty_mat = np.empty(mat_shape)
    empty_mat[:] = np.nan
    # Extract the montage:
    montage = raw.get_montage()
    # performing the laplace referencing:
    for ch in reference_mapping.keys():
        if reference_mapping[ch]["ref_1"] is None:
            ref_1 = empty_mat
        else:
            # Get the index of the reference channel:
            ind = data_chs.index(reference_mapping[ch]["ref_1"])
            ref_1 = np.squeeze(ref_data[ind, :])
        if reference_mapping[ch]["ref_2"] is None:
            ref_2 = empty_mat
        else:
            # Get the index of the reference channel:
            ind = data_chs.index(reference_mapping[ch]["ref_2"])
            ref_2 = np.squeeze(ref_data[ind, :])
        raw.apply_function(laplace_ref_fun, picks=[ch], n_jobs=n_jobs, verbose=None,
                           ref_1=ref_1, ref_2=ref_2,
                           channel_wise=True)
        # Relocating if needed:
        if relocate_edges:
            # If one of the two reference is only Nan, then there was one ref missing, in which case the channel must
            # be replaced, (bitwise or because only if one of the two is true)
            if np.isnan(ref_1).all() ^ np.isnan(ref_2).all():
                print("Relocating channel " + ch)
                montage = raw.get_montage()
                # Get the indices of the current channel
                ch_ind = montage.ch_names.index(ch)
                # Get the single reference:
                ref = reference_mapping[ch]["ref_1"] if reference_mapping[ch]["ref_1"] is not None \
                    else reference_mapping[ch]["ref_2"]
                ref_ind = montage.ch_names.index(ref)
                # Compute the center between the two:
                x, y, z = (montage.dig[ch_ind]["r"][0] + montage.dig[ref_ind]["r"][0]) / 2, \
                          (montage.dig[ch_ind]["r"][1] + montage.dig[ref_ind]["r"][1]) / 2, \
                          (montage.dig[ch_ind]["r"][2] + montage.dig[ref_ind]["r"][2]) / 2,
                montage.dig[ch_ind]["r"] = np.array([x, y, z])

    if relocate_edges:
        # Adding the montage back:
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")

    # Projecting the ecog channels to the surface if they were relocated:
    if relocate_edges:
        ecog_channels = mne.pick_types(raw.info, ecog=True)
        if len(ecog_channels) > 0:
            project_elec_to_surf(raw, subjects_dir, subject)

    return raw, reference_mapping, bad_channels


def baseline_scaling(epochs, correction_method="ratio", baseline=(None, 0), picks=None, n_jobs=1):
    """
    Perform baseline correction on the data.

    :param epochs: mne epochs object, epochs on which to perform the baseline correction
    :param correction_method: string, options to do the baseline correction
    :param baseline: tuple, which bit to take as the baseline
    :param picks: None or list of int or list of strings, indices or names of the channels on which to perform the correction
    :param n_jobs: int, number of jobs to use to preprocess the function
    :return: None
    """
    from mne.baseline import rescale
    epochs.apply_function(rescale, times=epochs.times, baseline=baseline, mode=correction_method,
                          picks=picks, n_jobs=n_jobs, )

    return None


def project_montage_to_surf(montage, channel_types, subject, fs_dir):
    """
    Project the electrodes to the pial surface. Note that this is only ever done on the ecog channels.

    :param montage: mne DigMontage object, montage to project
    :param channel_types: dict, dictionary specifying channel types
    :param subject: string, name of the subject
    :param fs_dir: path or string, path to the freesurfer directory
    :return: mne DigMontage object, updated montage
    """
    # Read the surfaces:
    left_surf = read_geometry(str(Path(fs_dir, subject, "surf", "lh.pial")))
    right_surf = read_geometry(str(Path(fs_dir, subject, "surf", "rh.pial")))
    # Extract the channels from the montage:
    ecog_channels = [ch for ch in montage.ch_names if channel_types[ch] == "ecog"]

    # Looping through each of these channels:
    for channel in ecog_channels:
        # Get the channel index
        ch_ind = montage.ch_names.index(channel)
        # Get the channel coordinates:
        ch_coord = montage.dig[ch_ind]["r"]
        # Checking if the coordinates are NAN:
        if math.isnan(ch_coord[0]):
            continue
        if ch_coord[0] < 0:
            # Compute x y and z distances to each vertex in the surface:
            b_x = np.absolute(left_surf[0][:, 0] - ch_coord[0] * 1000)
            b_y = np.absolute(left_surf[0][:, 1] - ch_coord[1] * 1000)
            b_z = np.absolute(left_surf[0][:, 2] - ch_coord[2] * 1000)
            # Find the shortest distance:
            d = np.sqrt(np.sum([np.square(b_x), np.square(b_y), np.square(b_z)], axis=0))
            # Get the index of the smallest one:
            min_vert_ind = np.argmin(d)
            montage.dig[ch_ind]["r"] = np.squeeze(np.array(
                [left_surf[0][min_vert_ind, 0] * 0.001, left_surf[0][min_vert_ind, 1] * 0.001,
                 left_surf[0][min_vert_ind, 2] * 0.001]))
        else:
            # Compute x y and z distances to each vertex in the surface:
            b_x = np.absolute(right_surf[0][:, 0] - ch_coord[0] * 1000)
            b_y = np.absolute(right_surf[0][:, 1] - ch_coord[1] * 1000)
            b_z = np.absolute(right_surf[0][:, 2] - ch_coord[2] * 1000)
            # Find the shortest distance:
            d = np.sqrt(np.sum([np.square(b_x), np.square(b_y), np.square(b_z)], axis=0))
            # Get the index of the smallest one:
            min_vert_ind = np.argmin(d)
            montage.dig[ch_ind]["r"] = np.squeeze(np.array(
                [right_surf[0][min_vert_ind, 0] * 0.001, right_surf[0][min_vert_ind, 1] * 0.001,
                 right_surf[0][min_vert_ind, 2] * 0.001]))

    return montage


def create_montage(channels, bids_path, fsaverage_dir):
    """
    Fetch the MNI coordinates of a set of channels from the bids directory directly. 

    :param channels: list of channel names to fetch MNI coordinates for. The name of the channels must follow the naming convention:
    $SUBID-$Channel_name, so CF102-G1 for example
    :param bids_path: mne_bids BIDSPath object with information to fetch coordinates
    :param fsaverage_dir: path to the FreeSurfer root folder containing the fsaverage
    :return: mne.Info object with channel info, including positions in MNI space
    """

    # Extract unique subjects from the channels list
    subjects = list(set(channel.split('-')[0] for channel in channels))

    # Initialize dictionaries to store MNI coordinates and channel types
    mni_coords = {}
    channels_types = {}

    for subject in subjects:
        # Extract channels for the current subject
        subject_channels = [channel.split('-')[1] for channel in channels if channel.split('-')[0] == subject]

        # Create the path to the subject's BIDS directory
        subject_path = BIDSPath(root=bids_path.root, subject=subject,
                                session=bids_path.session, datatype=bids_path.datatype, task=bids_path.task)

        # Define the filenames for coordinates and channels
        coordinates_file = f'sub-{subject}_ses-{subject_path.session}_space-fsaverage_electrodes.tsv'
        channel_file = f'sub-{subject}_ses-{subject_path.session}_task-{bids_path.task}_channels.tsv'

        # Load the coordinates and channels data
        coordinates_df = pd.read_csv(Path(subject_path.directory, coordinates_file), sep='\t')
        channels_df = pd.read_csv(Path(subject_path.directory, channel_file), sep='\t')

        # Extract the types of the channels
        subject_channel_type = (channels_df.loc[channels_df['name'].isin(subject_channels), ['name', 'type']]
                                .set_index('name').to_dict())['type']
        subject_channel_type = {f"{subject}-{ch}": subject_channel_type[ch] for ch in subject_channel_type}
        channels_types.update(subject_channel_type)

        # Extract the MNI coordinates
        for ch in subject_channels:
            mni_coords[f"{subject}-{ch}"] = np.squeeze(
                coordinates_df.loc[coordinates_df['name'] == ch, ['x', 'y', 'z']].to_numpy())

    # Create the montage
    montage = mne.channels.make_dig_montage(ch_pos=mni_coords, coord_frame='mni_tal')

    # Ensure channel types are in lowercase
    channels_types = {ch: channels_types[ch].lower() for ch in channels_types}

    # Add the MNI fiducials
    montage.add_mni_fiducials(fsaverage_dir)

    # Create the info object with channel names and types
    info = mne.create_info(ch_names=channels, ch_types=list(channels_types.values()), sfreq=100)

    # Set the montage for the info object
    info.set_montage(montage)

    return info


def get_roi_channels(channels, rois, bids_path, atlas):
    """
    Get channels that are in a particular set of ROIs. This functions relies on the labels of each channel being found
    in csv table located in the bids directory of each single participant.

    :param channels: list of string, list of channels
    :param rois: list of strings, list of ROIs with names matching the labels of a particular atlas
    :param bids_path: mne bids path object
    :param atlas: string, name of the atlas of interest
    :return: list of string, list of channels found within the region of interest
    """

    # Load the atlas of that particular subject:
    atlas_file = Path(bids_path.root,
                      'sub-' + bids_path.subject, 'ses-' + bids_path.session, bids_path.datatype,
                      f'sub-{bids_path.subject}_ses-{bids_path.session}_atlas-{atlas}_labels.tsv')
    atlas_df = pd.read_csv(atlas_file, sep='\t')
    # Loop through each channel:
    roi_channels = []
    for channel in channels:
        # Extract channels rois:
        try:
            channel_labels = atlas_df.loc[atlas_df['channel'] == channel, 'region'].values[0].split('/')
        except AttributeError:
            print('WARNING: No loc for {}'.format(channel))
            continue
        except IndexError:
            print('WARNING: {} not found in channels localization file'.format(channel))
        # Remove ctx l and r
        channel_labels = [lbl.replace('ctx_lh_', '').replace('ctx_rh_', '') for lbl in channel_labels]
        # Check whether any of the channel label is within the list of rois:
        for ch_lbl in channel_labels:
            if ch_lbl in rois:
                roi_channels.append(channel)
                break

    return roi_channels


def mri_2_mni(montage, subject, fs_dir):
    """
    Convert MRI coordinates to MNI coordinates using the Talairach transform.

    :param montage: mne DigMontage object
    :param subject: string, subject ID
    :param fs_dir: path or string, path to the freesurfer directory
    :return: dict, MNI positions
    """
    # Add the fiducials: ras -> mri:
    # montage, trans = add_fiducials(montage, fs_dir, subject)

    # Fetch the transform from mri -> mni
    mri_mni_trans = mne.read_talxfm(subject, fs_dir)
    # Apply the transform mri -> fsaverage:
    montage.apply_trans(mri_mni_trans)  # mri to mni_tal (MNI Taliarach)
    # Transform mni_tal back to mri coordinates (with identity matrix) since we want to plot in mri coordinates:
    montage.apply_trans(mne.transforms.Transform(fro="mni_tal", to="mri", trans=np.eye(4)))
    mni_positions = montage.get_positions()['ch_pos']
    return mni_positions


def count_colors(values, cmap):
    """
    Count the number of entries for each value and map them to colors.

    :param values: list of values
    :param cmap: string, name of the colormap
    :return: dict, mapping keys to RGB colors
    """
    from collections import Counter
    # Count the number of entries for each values:
    counts = dict(Counter(values))

    # Get the list of counts from the dictionary
    counts_val = list(counts.values())

    # Create a colormap
    cmap = plt.cm.get_cmap(cmap)

    # Normalize the counts to [0, 1] to match the colormap's scale
    norm_counts = [(count - min(counts_val)) / (max(counts_val) - min(counts_val)) for count in counts_val]

    # Map normalized counts to colors
    colors = [cmap(norm) for norm in norm_counts]

    # Convert RGBA colors to RGB
    rgb_colors = [(r, g, b) for r, g, b, _ in colors]

    # Create a new dictionary mapping keys to RGB colors
    val_colors = dict(zip(counts.keys(), rgb_colors))

    return val_colors


def exclude_distant_channels(montage, subject, fs_dir, max_dist=5):
    """
    Exclude electrodes that are further than max_dist from the brain surface.

    :param montage: mne DigMontage object
    :param subject: string, subject ID
    :param fs_dir: path or string, path to the freesurfer directory
    :param max_dist: float, maximum distance from the brain surface
    :return: mne DigMontage object, updated montage
    """
    # Load the surface file using nibabel
    # Create the file names to the directory:
    lhfile = Path(fs_dir, subject, "surf", "lh.pial")
    rhfile = Path(fs_dir, subject, "surf", "rh.pial")
    left_surf = read_geometry(str(lhfile))
    right_surf = read_geometry(str(rhfile))
    # Combine all vertices:
    full_surf = np.concatenate([left_surf[0], right_surf[0]])
    out_elec = []

    # Loop through each electrode in the montage:
    ch_positions = montage.get_positions()['ch_pos']
    for ch, pos in ch_positions.items():
        dist = np.sqrt(np.sum((full_surf - pos * 1000) ** 2, axis=1))
        if np.min(dist) >= max_dist:
            # Find the index of this electrode:
            ch_ind = montage.ch_names.index(ch)
            del montage.ch_names[ch_ind]
            del montage.dig[ch_ind]
    return montage


def plot_ieeg_image(epo, channel, order=None, show=False, units="HGP (norm.)", scalings=1, cmap="RdYlBu_r",
                    center=1, ylim_prctile=95, logistic_cmap=True, ci=0.95, evk_method="mean", ci_method="mean",
                    evk_colors="k"):
    """
    Plot an iEEG (intracranial EEG) image with the option to use logistic normalization for the colormap,
    including confidence intervals and evoked responses.

    Parameters:
    -----------
    epo : mne.Epochs
        The MNE Epochs object containing the data to be plotted.
    channel : str
        The name of the channel to plot.
    order : array-like | None
        If not None, reorder the images by this array. Default is None.
    show : bool, optional
        Whether to display the figure immediately. Default is False.
    units : str, optional
        The unit label for the y-axis. Default is "HGP (norm.)".
    scalings : float, optional
        Scaling factor for the data. Default is 1.
    cmap : str or None, optional
        Colormap for the image. Default is "RdYlBu_r". If None, uses the default colormap.
    center : float or None, optional
        Center value for the color normalization (used for symmetric data). Default is 1.
        If None, no centering is applied. Is overriden by logistic_cmap if activated
    ylim_prctile : float or list of floats, optional
        Percentile(s) to define the y-axis limits. Default is 95.
        If an integer is provided, it defines both upper and lower limits symmetrically.
    logistic_cmap : bool, optional
        Whether to apply logistic mapping to the colormap to emphasize the data's dynamic range.
        Default is True.
    ci : float, optional
        Confidence interval percentage. Default is 0.95 (95% confidence interval).
    evk_method : str, optional
        Method to compute the evoked response. Default is "mean".
    ci_method : str, optional
        Method to compute the confidence interval. Default is "mean".
    evk_colors : str, optional
        Color for the evoked response plot. Default is "k" (black).

    Returns:
    --------
    figs : list
        A list of figures generated by the plot.
    """

    # Define the logistic forward and inverse functions
    def _forward(x, x0=0, k=1):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def _inverse(y, x0=0, k=1):
        return x0 - (1 / k) * np.log((1 / y) - 1)

    # Create the colormap:
    if cmap is None:
        cmap = plt.get_cmap("RdYlBu_r")
    else:
        cmap = plt.get_cmap(cmap)

    # Get the y limits:
    data = np.squeeze(epo.get_data(picks=channel))
    if ylim_prctile is None:
        vmax = np.max(data)
        vmin = np.min(data)
    else:
        if isinstance(ylim_prctile, int):
            ylim_prctile = [ylim_prctile, 100 - ylim_prctile]

        vmax = np.percentile(data, ylim_prctile[0])
        vmin = np.percentile(data, ylim_prctile[1])

    # Apply centered normalization if `center` is provided
    if center is not None:
        norm = matplotlib.colors.CenteredNorm(vcenter=center)
        cmap = mcolors.ListedColormap(cmap(norm(np.linspace(vmin, vmax, 256))))

    # Apply logistic normalization if enabled
    if logistic_cmap:
        norm = matplotlib.colors.FuncNorm((lambda x: _forward(x, x0=0, k=1),
                                           lambda y: _inverse(y, x0=0, k=1)),
                                          vmin=vmin, vmax=vmax)
        cmap = mcolors.ListedColormap(cmap(norm(np.linspace(vmin, vmax, 256))))

    # Plot the image:
    figs = mne.viz.plot_epochs_image(epo, picks=channel, show=show, order=order,
                                     units=dict(ecog=units, seeg=units),
                                     scalings=dict(ecog=scalings, seeg=scalings),
                                     evoked=True, cmap=cmap, vmin=vmin, vmax=vmax)

    # Compute the evoked response:
    evk = np.squeeze(epo.average(picks=channel, method=evk_method).get_data())

    # Compute the confidence interval:
    ci_low, ci_up = mne.stats.bootstrap_confidence_interval(epo.get_data(picks=channel),
                                                            stat_fun=ci_method, ci=ci)

    # Clear the existing axis and replace it:
    ax = figs[0].axes[1]
    ax.cla()
    ax.plot(epo.times, evk, color=evk_colors)
    ax.fill_between(epo.times, np.squeeze(ci_up), np.squeeze(ci_low), color=evk_colors, alpha=0.3)
    ax.set_xlim([epo.times[0], epo.times[-1]])
    ax.set_ylabel(units)
    ax.set_xlabel("Time (s)")

    return figs
