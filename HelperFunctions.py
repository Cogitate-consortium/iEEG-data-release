""" This script contains all helper function for the preprocessing pipeline
    authors: Alex Lepauvre and Katarina Bendtz
    alex.lepauvre@ae.mpg.de
    katarina.bendtz@tch.harvard.edu
    Dec 2020
"""

import os
from pathlib import Path
import numpy as np
import math

import mne
from mne.viz import plot_alignment, snapshot_brain_montage
from mne.datasets import fetch_fsaverage
from nibabel.freesurfer.io import read_geometry

import matplotlib.pyplot as plt

import pandas as pd

import json


def path_generator(directory):
    """
    This function generates a folder if it doesn't exists
    :param directory: (string or pathlib path object) folder to create
    :return:
    """
    #
    if not os.path.isdir(directory):
        # Creating the directory:
        os.makedirs(directory)

    return directory


def create_dig_montage(mne_data, bids_path, montage_space='T1'):
    """
    This function adds a dig montage to the mne data, from the space specified. This is required as in our csae, we
    have both the fsaverage and T1 montages within the BIDS root, which MNE bids doesn't handle
    :param mne_data: (mne data object, raw, epochs or evoked) object to which the montage should be added
    :param bids_path: (bids path object from mne_bids) contains paths information
    :param montage_space: (string) T1 for subject native T1 space, MNI for fsaverage space
    :return:
    """
    # Handle montage type
    if montage_space.upper() == "T1":
        space = "Other"
        coord_frame = "mri"
    elif montage_space.upper() == "MNI":
        space = "fsaverage"
        coord_frame = "mni_tal"
    else:
        raise Exception("You have passed a montage space that is not supported. It should be either T1 or"
                        "MNI! Check your config")

    # Create channels localization file:
    channels_loc = Path(bids_path.directory,
                        'sub-{}_ses-{}_space-{}_electrodes.tsv'.format(bids_path.subject, bids_path.session,
                                                                       space))
    # Load the file:
    channels_coordinates = pd.read_csv(channels_loc, sep='\t')
    # From this file, getting the channels:
    channels = channels_coordinates["name"].tolist()
    # Get the position:
    position = channels_coordinates[["x", "y", "z"]].to_numpy()
    # Create the montage:
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(channels, position)),
                                            coord_frame=coord_frame)
    # And set the montage on the raw object:
    mne_data.set_montage(montage, on_missing='warn')

    return mne_data


def save_config(config, save_path, step, signal, file_prefix, file_extension):
    """
    This function saves the configs that was used to generate the data. It saves both the entire config file and the one
    specific to the particular step
    :param config: (dictionary) copy of the parameters used to generate the data. Always saving
    it alongside the data to always know what happened to them
    :param save_path: (path string of path object) path to where the data should be saved
    :param step: (string) name of the preprocessing step, saving the data in a separate folder
    :param signal: (string) name of the signal we are saving
    :param file_prefix: (string) prefix of the file to save. We follow the convention to have
    the subject ID, the session and the task.
    :param file_extension: (string) file name extension
    """
    # Saving the config of this particular step:
    config_file_name = Path(save_path, '{}_desc-{}_ieeg{}.json'.format(file_prefix, step,
                                                                       file_extension.split('.')[0]))
    with open(str(config_file_name), 'w') as outfile:
        json.dump(config[step][signal], outfile)
    # Saving the entire config file as well:
    config_file_name = Path(save_path, '{}_desc-{}_ieeg{}.json'.format(file_prefix, 'all',
                                                                       file_extension.split('.')[0]))
    with open(str(config_file_name), 'w') as outfile:
        json.dump(config, outfile)

    return None


def mne_data_saver(data, config, save_root, step, signal, file_prefix,
                   file_extension="-raw.fif"):
    """
    This function saves the different instances of mne objects
    :param data: (mne object: epochs, evoked, raw...) data to be saved
    :param config: (dictionary) copy of the parameters used to generate the data. Always saving
    it alongside the data to always know what happened to them
    :param save_root: (path string of path object) path to where the data should be saved
    :param step: (string) name of the preprocessing step, saving the data in a separate folder
    :param signal: (string) name of the signal we are saving
    :param file_prefix: (string) prefix of the file to save. We follow the convention to have
    the subject ID, the session and the task.
    :param file_extension: (string) file name extension
    :return:
    """
    print("=" * 40)
    print("Saving mne object")

    # First, generating the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)
    # Generating the full file name:
    full_file_name = Path(save_path, '{}_desc-{}_ieeg{}'.format(file_prefix, step, file_extension))
    # Saving the data:
    data.save(full_file_name, overwrite=True)
    # Saving the config:
    save_config(config, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return None


def plot_channels_psd(raw, save_root, step, signal, file_prefix,
                      file_extension="psd.png", plot_single_channels=False, channels_type=None):
    """
    This function plots and saved the psd of the chosen electrodes. There is also the option to plot each channel
    separately
    :param raw: (mne raw object)
    :param save_root: (path string of path object) path to where the data should be saved
    :param step: (string) name of the preprocessing step, saving the data in a separate folder
    :param signal: (string) name of the signal we are saving
    :param file_prefix: (string) prefix of the file to save. We follow the convention to have
    the subject ID, the session and the task.
    :param file_extension: (string) file name extension
    :param plot_single_channels: (boolean) whether to plot single channels or only all of
    them superimposed
    :param channels_type: (dict) list of the channels of interest
    :return:
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
        # Compute the PSD for all the picks:
        psd, freqs = mne.time_frequency.psd_welch(raw, picks=picks, average="mean")
        for ind, pick in enumerate(picks):
            fig, ax = plt.subplots(figsize=[15, 6])
            ax.plot(freqs, np.log(psd[np.arange(psd.shape[0]) != ind, :].T), linewidth=0.5, color="k", alpha=0.8)
            ax.plot(freqs, np.log(psd[ind, :].T).T, linewidth=2, color="r")
            ax.set_ylabel("\u03BCV\u00B2/Hz (dB)")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_xlim([freqs[0], freqs[-1]])
            ax.grid(linestyle=":")
            ax.set_title("{} PSD".format(raw.ch_names[pick]))
            full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}{}'.format(file_prefix, step,
                                                                           raw.ch_names[pick], file_extension))
            plt.savefig(full_file_name, dpi=300, transparent=True)
            plt.close()

    return None


def plot_bad_channels(raw, save_root, step, signal, file_prefix,
                      file_extension="bads.png", plot_single_channels=False, picks="bads"):
    """
    This function plots the bad channels psd and raw signal to show what it being disarded:
    :param raw: (mne raw object)
    :param save_root: (path string of path object) path to where the data should be saved
    :param step: (string) name of the preprocessing step, saving the data in a separate folder
    :param signal: (string) name of the signal we are saving
    :param file_prefix: (string) prefix of the file to save. We follow the convention to have
    the subject ID, the session and the task.
    :param file_extension: (string) file name extension
    :param plot_single_channels: (boolean) whether or not to plot single channels or only all of them superimposed
    :param picks: (list) list of the channels of interest
    :return:
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
    This function takes specific channel types as reference and averages the amplitude across them along time. This mean
    time series is then substract from, all the channels at each time points.
    :param raw: (mne raw object) contains the data to be rereferenced
    :param reference_channel_types: (dict) dictionary specifying the channels types to be take as reference, as well as
    which to exclude. See mne documentation for mne.pick_types to know what you can pass here
    :param target_channel_types: (dict) dictionary specifying the channels types to apply reference to, as well as
    which to exclude. See mne documentation for mne.pick_types to know what you can pass here
    :return: raw: (mne raw object) modified instance of the mne object. Note that the data are modified in place, so
    use copy upon calling this function to avoid overwriting things
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
    This function filters the raw data according to the set parameters
    :param raw: (mne raw object) continuous data
    :param njobs: (int) number of jobs to preprocessing the filtering in parallel threads
    :param frequency: (int or float) frequency to notch out.
    :param remove_harmonics: (boolean) whether or not to remove all the harmonics of the declared freq (up until the
    sampling freq)
    :param filter_type: (string) what type of filter to use, iir or fir
    :param cutoff_lowpass_bw: (float) frequency for low pass (only used if type is iir)
    :param cutoff_highpass_bw: (float) frequency for high pass (only used if type is iir)
    :param channel_types: (dict) type of channels to notch filter, boolean for the channel types
    :return:
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
    This function parses the events found in the epochs descriptions to create the meta data. The column of the meta
    data are generated based on the metadata column names. The column name must be a list in the same order as the
    strings describing the events. The name of the column must be the name of the overall condition, so say the
    specific column describes the category of the presented stim (faces, objects...), then the column should be called
    category. This will become obsolete here at some point, when the preprocessing is changed to generate the meta data
    directly
    :param epochs: (mne epochs object) epochs for which the meta data will be generated
    :param metadata_column_names: (list of strings) name of the column of the meta data. Must be in the same order
    as the events description + must be of the same length as the number of word in the events description
    :return: epochs (mne epochs object)
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
    This function performs the epoching according to a few parameters
    :param raw: (mne raw object) data to epochs
    :param events: (mne events numpy array) three cols, one for event time stamp, the other for the event ID. This
    is what you get out from the mne functions to extract the events
    :param events_dict: (dict) mapping between the events ID and their descriptions
    :param picks: (string or list of int or list of strings) channels to epochs. If you pass all, it will select all the
    channels found in the raw object
    :param tmin: (float) how long before each event of interest to epochs. IN SECONDS!!!
    :param tmax: (float) how long before each event of interest to epochs. IN SECONDS!!!
    :param events_not_to_epoch: (list of strings) name of the events not to epochs. This is more handy than passing
    all the ones we want to epochs, because usually there are more events you are interested about than events you are
    not interested about
    :param baseline: (tuple of floats) passed to the mne epoching function to apply baseline correction and what is
    defined as the baseline
    :param reject_by_annotation: (boolean) whether or not to discard trials based on annotations that were made
    previously
    :param meta_data_column: (list of strings) list of the column names of hte metadata. The metadata are generated
    by parsing the extensive trial description you might have as events. For example, if you have something like this:
    Face/short/left, the meta data generator will create a table:
    col1  | col2  | col3
    Face    short   left
    The list you pass is the name of the different columns. So you would pass: ["category", "duration", "position"]:
    category  | duration  | position
    Face         short        left
    :return:
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
                        tmax=tmax, baseline=baseline, verbose='ERROR', picks=picks,
                        reject_by_annotation=reject_by_annotation)
    # Dropping the bad epochs if there were any:
    epochs.drop_bad()
    # Adding the meta data to the table. The meta data are created by parsing the events strings, as each substring
    # contains specific info about the trial:
    if meta_data_column is not None:
        epochs = create_metadata_from_events(epochs, meta_data_column)

    return epochs


def automated_artifact_detection(epochs, standard_deviation_cutoff=4, trial_proportion_cutoff=0.1, channel_types=None,
                                 aggregation_function="peak_to_peak"):
    """
    This function perform a basic artifact rejection from the each electrode separately
    and then also removes the trial from all eletrodes if it is considered an outlier
    for more than X percent of the electrodes. NOTE: THIS FUNCTION DOES NOT DISCARD THE TRIALS THAT ARE DETECTED AS
    BEING BAD!!!
    :param epochs: (mne epochs object) epochs for which to reject trials.
    :param standard_deviation_cutoff: (float or int) number of standard deviation the amplitude of the data  a given
    trial must be to be considered an outlier
    :param trial_proportion_cutoff: (int between 0 and 1) proportion of channels for which the given trial must be
    considered "bad" (as defined by std factor) for a trial to be discarded all together
    :param channel_types: (dict: "chan_type": True) dictionary containing a boolean for each channel type to select
    which channels are considered by this step
    :param aggregation_function: (string) name of the function used to aggregate the data within trials and channels
    across time. If you set "mean", the mean will be computed within trial and channel. The standard deviation thereof
    will be used to compute the rejection theshold
    :return:
    """
    # Selecting electrodes of interest:
    if channel_types is None:
        channel_types = {"seeg": True, "ecog": True}
    picks = mne.pick_types(epochs.info, **channel_types)

    if aggregation_function == "peak_to_peak" or aggregation_function == "ptp":
        spec_func = np.ptp
    elif aggregation_function == "mean" or aggregation_function == "average":
        spec_func = np.mean
    elif aggregation_function == "auc" or aggregation_function == "area_under_the_curve":
        spec_func = np.trapz
    else:
        raise Exception("You have passed a function for aggregation that is not support! The argument "
                        "\naggregation_function must be one of the following:"
                        "\npeak_to_peak"
                        "\nmean"
                        "\nauc")
    # ------------------------------------------------------------------------------------------------------------------
    # Computing thresholds and rejecting trials based on it:
    # Compute the aggregation (average, ptp, auc...) within trial and channel across time:
    trial_aggreg = spec_func(epochs.get_data(picks=picks), axis=2)
    # Computing the standard deviation across trials but still within electrode of the aggregation measure:
    stdev = np.std(trial_aggreg, axis=0)
    # Computing the average of the dependent measure across trials within channel:
    average = np.mean(trial_aggreg, axis=0)
    # Computing the artifact boundaries (beyond which a trial will be considered artifactual) by taking the mean +- n
    # times the std of the aggregated meaasure:
    artifact_thresh = np.array([average - stdev * standard_deviation_cutoff,
                                average + stdev * standard_deviation_cutoff])
    print(epochs.ch_names)
    print(artifact_thresh)
    # Comparing the values of each trial of a given channel against the threshold of that specific channel. Boolean
    # outcome:
    rejection_matrix = \
        np.array([np.array((trial_aggreg[:, i] > artifact_thresh[1, i]) & (trial_aggreg[:, i] > artifact_thresh[0, i]))
                  for i in range(trial_aggreg.shape[1])]).astype(float)
    # trial_proportion_cutoff dictates the proportion of channel for whom trial must be marked as bad to discard that
    # specific trial across all channels. Averaging the rejection matrix across channels to reject specific trials:
    ind_trials_to_drop = np.where(
        np.mean(rejection_matrix, axis=0) > trial_proportion_cutoff)[0]

    return epochs, ind_trials_to_drop


def find_interruption_index(events, event_dict, interuption_landmark, interruption_block_num=None):
    """
    This function asks the user whether there was an interruption in the experiment. If there was one, the function
    will ask which block the interruption occured in to find the index of the interruption. This is required by some
    functions that compute averaging accross the experiment. Indeed, if you had interruption, the signal change a lot
    inbetween, and averaging across the experiment without regard for the interruption is meaningless.
    :param events: (np array) contain the events indices and their identity
    :param event_dict: (dict) mapping between the events identity and their description
    :param interuption_landmark: (string) name of what to use to refer to for the interruption findings.
    :param interruption_block_num: (None or int) if you already know when the interruption occured, it can be passed
    in this function to avoid passing things manually.
    :return: interruption_index: int, index of where the interruption occured.
    """

    # Asking whether there was an interruption. If the interruption block number is not none, then no need to ask:
    if interruption_block_num is None:
        interuption = \
            input("Was your experiment interrupted at some point?")
    else:
        interuption = "yes"

    if interuption.lower() == "yes":
        if interruption_block_num is None:
            interruption_block_num = \
                input("In which {0} was your experiment interrupted?".format(
                    interuption_landmark))
        else:
            interruption_block_num = str(interruption_block_num)
        # Creating the description string for when the interruption happened:
        interuption_event_desc = interuption_landmark + "_" + interruption_block_num
        # Looking for all events fitting the description:
        evts = [event_dict[desc]
                for desc in event_dict if interuption_event_desc in desc]
        # Now looking for the index of the earliest occurence:
        interruption_index = int(min([evt[0] for evt in events if evt[2] in evts]))
    else:
        interruption_index = False

    return interruption_index


def compute_hg(raw, frequency_range=None, njobs=1, bands_width=10, channel_types=None,
               do_baseline_normalization=True):
    """
    This function computes the high gamma signal by filtering the signal in bins between a defined frequency range.
    In each frequency bin, the hilbert transform is applied and the amplitude is extracted (i.e. envelope). The envelope
    is then normalized by dividing by the mean amplitude over the entire time series (within each channel separately).
    The amplitude is then averaged across frequency bins. This follows the approach described here:
    https://www.nature.com/articles/s41467-019-12623-6#Sec10
    This function computes the envelope in specified frequency band. It further has the option to compute envelope
    in specified bands within the passed frequency to then do baseline normalization to account for 1/f noise.
    :param raw: (mne raw object) raw mne object containing our raw signal
    :param frequency_range: (list of list of floats) frequency of interest
    :param bands_width: (float or int) width of the frequency bands to loop overs
    :param njobs: (int) number of parallel processes to compute the high gammas
    :param channel_types: (dict) name of the channels for which the high gamma should be computed. This is important
    to avoid taking in electrodes which we might not want
    :param do_baseline_normalization: (bool) whether to do baseline normalization
    :return: frequency_band_signal: (mne raw object) dictionary containing raw objects with high gamma in the different
    frequency bands
    """

    def divide_by_average(data):
        print('Dividing channels amplitude by average amplitude')
        if not isinstance(data, np.ndarray):
            raise TypeError('Input value must be an ndarray')
        if data.ndim != 2:
            raise TypeError('The data should have two dimensions!')
        else:
            norm_data = data / data.mean(axis=1)[:, None]
            # Check that the normalized data are indeed what they should be. We are dividing the data by a constant.
            # So if we divide the normalized data by the original data, we should get a constant:
            if len(np.unique((data[0, :] / norm_data[0, :]).round(decimals=15))) != 1:
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
        print('')
        print('-' * 40)
        print('Computing the frequency in band: ' + str(freq_bin))
        # Filtering the signal and apply the hilbert:
        print('Computing the envelope amplitude')
        bin_power = freq_band_raw.copy().filter(freq_bin[0], freq_bin[1],
                                                n_jobs=njobs).apply_hilbert(envelope=True)

        # Now, dividing the amplitude by the mean, channel wise:
        if do_baseline_normalization:
            print('Divide by average')
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
    The erp computation consist in low passing the raw signal to extract only low freq signal.
    :param raw: (mne raw object) raw mne object containing our raw signal
    :param frequency_range: (list of list of floats) frequency of interest
    :param njobs: (int) number of parallel processes to compute the high gammas
    :param channel_types: (dict) type of channel for which the ERP should be computed. It should be of the format:
    {ecog: True, seeg: True...}
    :param kwargs: arguments that can be passed to the mne raw.filter function. Check this page to find the options:
    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter
    :return: erp_raw: (mne raw object) computed erp signal
    frequency bands
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


def add_fiducials(raw, fs_directory, subject_id):
    """
    This function add the estimated fiducials to the montage and compute the transformation
    :param raw: (mne raw object) data to which the fiducials should be added
    :param fs_directory: (path string) path to the freesurfer directory
    :param subject_id: (string) name of the subject
    :return: mne raw object and transformation
    """
    montage = raw.get_montage()
    # If the coordinates are in mni_tal coordinates:
    if montage.get_positions()['coord_frame'] == "mni_tal":
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = Path(sample_path, 'subjects')
        montage.add_mni_fiducials(subjects_dir)
        trans = 'fsaverage'
    else:
        montage.add_estimated_fiducials(subject_id, fs_directory)
        trans = mne.channels.compute_native_head_t(montage)
    raw.set_montage(montage, on_missing="warn")

    return raw, trans


def plot_electrode_localization(mne_object, subject, fs_dir, config, save_root, step, signal, file_prefix,
                                montage_space="T1", file_extension='-loc.png', channels_to_plot=None,
                                plot_elec_name=False):
    """
    This function plots and saved the psd of the chosen electrodes.
    :param mne_object: (mne object: raw, epochs, evoked...) contains the mne object with the channels info
    :param subject_info: (custom object) contains info about the subject
    :param preprocessing_parameters: (custom object) contains the preprocessing info, required to generate the
    save directory
    :param step_name: (string) name of the step that this is performed under to save the data
    :param subject_id: (string) name of the subject! Not necessary if you want to plot in mni space
    :param fs_subjects_directory: (string or pathlib path) path to the free surfer subjects directory. Not required if
    you want to plot in mni space
    :param data_type: (string) type of data that are being plotted
    :param file_extension: (string) extension of the pic file for saving
    :param channels_to_plot: (list) contains the different channels to plot. Can pass list of channels types, channels
    indices, channels names...
    :param montage_space: (string)
    :param plot_elec_name: (string) whethre or not to print the electrodes names onto the snapshot!
    :return:
    """
    if channels_to_plot is None:
        channels_to_plot = ["ecog", "seeg"]
    # First, generating the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)

    # Adding the estimated fiducials and compute the transformation to head:
    mne_object, trans = add_fiducials(mne_object, fs_dir, subject)
    # If we are plotting the electrodes in MNI space, fetching the data:
    if montage_space.lower() == "mni":
        fs_subjects_directory = mne.datasets.sample.data_path() + '/subjects'
        subject = "fsaverage"
        fetch_fsaverage(subjects_dir=fs_subjects_directory, verbose=True)

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
            mne.utils.warn("You have attempted to plot {0} channels, but none where found in your signal".
                           format(ch_type),
                           RuntimeWarning)
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
                full_file_name = Path(save_path, '{}_desc-{}_ieeg_view-{}_names{}'.format(file_prefix, step,
                                                                                          ori, file_extension))
            else:
                full_file_name = Path(save_path, '{}_desc-{}_ieeg_view-{}{}'.format(file_prefix, step, ori,
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
    save_config(config, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return None


def get_montage_volume_labels_wang(montage, subject, subjects_dir=None,
                                   aseg='wang15_mplbl', dist=2):
    """Get regions of interest near channels from a Freesurfer parcellation.
    .. note:: This is applicable for channels inside the brain
              (intracranial electrodes).
    Parameters
    ----------
    %(montage)s
    %(subject)s
    %(subjects_dir)s
    %(aseg)s
    dist : float
        The distance in mm to use for identifying regions of interest.
    Returns
    -------
    labels : dict
        The regions of interest labels within ``dist`` of each channel.
    colors : dict
        The Freesurfer lookup table colors for the labels.
    """
    import numpy as np
    import os.path as op
    from mne.channels import DigMontage
    from mne._freesurfer import read_freesurfer_lut
    from mne.utils import get_subjects_dir, _check_fname, _validate_type
    from mne.transforms import apply_trans
    from mne.surface import _voxel_neighbors, _VOXELS_MAX
    from collections import OrderedDict

    _validate_type(montage, DigMontage, 'montage')
    _validate_type(dist, (int, float), 'dist')

    if dist < 0 or dist > 10:
        raise ValueError('`dist` must be between 0 and 10')

    import nibabel as nib
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    aseg = _check_fname(op.join(subjects_dir, subject, 'mri', aseg + '.mgz'),
                        overwrite='read', must_exist=True)
    aseg = nib.load(aseg)
    aseg_data = np.array(aseg.dataobj)

    # read freesurfer lookup table
    lut, fs_colors = read_freesurfer_lut(
        op.join(subjects_dir, 'wang2015_LUT.txt'))  # put the wang2015_LUT.txt into the freesurfer subjects dir
    label_lut = {v: k for k, v in lut.items()}

    # assert that all the values in the aseg are in the labels
    assert all([idx in label_lut for idx in np.unique(aseg_data)])

    # get transform to surface RAS for distance units instead of voxels
    vox2ras_tkr = aseg.header.get_vox2ras_tkr()

    ch_dict = montage.get_positions()
    if ch_dict['coord_frame'] != 'mri':
        raise RuntimeError('Coordinate frame not supported, expected '
                           '"mri", got ' + str(ch_dict['coord_frame']))
    ch_coords = np.array(list(ch_dict['ch_pos'].values()))

    # convert to freesurfer voxel space
    ch_coords = apply_trans(
        np.linalg.inv(aseg.header.get_vox2ras_tkr()), ch_coords * 1000)
    labels = OrderedDict()
    for ch_name, ch_coord in zip(montage.ch_names, ch_coords):
        if np.isnan(ch_coord).any():
            labels[ch_name] = list()
        else:
            voxels = _voxel_neighbors(
                ch_coord, aseg_data, dist=dist, vox2ras_tkr=vox2ras_tkr,
                voxels_max=_VOXELS_MAX)
            label_idxs = set([aseg_data[tuple(voxel)].astype(int)
                              for voxel in voxels])
            labels[ch_name] = [label_lut[idx] for idx in label_idxs]

    all_labels = set([label for val in labels.values() for label in val])
    colors = {label: tuple(fs_colors[label][:3] / 255) + (1.,)
              for label in all_labels}
    return labels, colors


def roi_mapping(mne_object, list_parcellations, subject, fs_dir, config, save_root, step, signal, file_prefix,
                file_extension='mapping.csv'):
    """
    This function maps the electrodes on different atlases. You can pass whatever atlas you have the corresponding
    free surfer parcellation for.
    :param mne_object: (mne raw or epochs object) object containing the montage to extract the roi
    :param list_parcellations: (list of string) list of the parcellation files to use to do the mapping. Must match the
    naming of the free surfer parcellation files.
    :param subject: (string) name of the subject to do access the fs recon
    :param fs_dir: (string or pathlib path object) path to the freesurfer directory containing all the participants
    :param config: (dictionary) copy of the parameters used to generate the data. Always saving
    it alongside the data to always know what happened to them
    :param save_root: (path string of path object) path to where the data should be saved
    :param step: (string) name of the preprocessing step, saving the data in a separate folder
    :param signal: (string) name of the signal we are saving
    :param file_prefix: (string) prefix of the file to save. We follow the convention to have
    the subject ID, the session and the task.
    :param file_extension: (string) file name extension
    :return: labels_df: (dict of dataframe) one data frame per parcellation with the mapping between roi and channels
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
            if parcellation == "wang15_mplbl":
                labels, _ = get_montage_volume_labels_wang(montage, "fsaverage", subjects_dir=None,
                                                           aseg='wang15_mplbl', dist=2)
            else:
                labels, _ = \
                    mne.get_montage_volume_labels(montage, "fsaverage", subjects_dir=subjects_dir, aseg=parcellation)
        else:
            if parcellation == "wang15_mplbl":
                labels, _ = get_montage_volume_labels_wang(mne_object.get_montage(), subject, subjects_dir=fs_dir,
                                                           aseg=parcellation, dist=2)
            else:
                labels, _ = mne.get_montage_volume_labels(
                    mne_object.get_montage(), subject, subjects_dir=fs_dir, aseg=parcellation)
        # Appending the electrodes roi to the table:
        for ind, channel in enumerate(labels.keys()):
            labels_df[parcellation] = labels_df[parcellation].append(
                pd.DataFrame({"channel": channel, "region": "/".join(labels[channel])}, index=[ind]))
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
    save_config(config, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return labels_df


def description_ch_rejection(raw, bids_path, channels_description, discard_bads=True):
    """
    This function enables to discard channels based on the descriptions found in the _channel.tsv file in the bids.
    A string or list of strings must be passed to be compared to the content of the _channel file to discard those
    matching
    :param raw: (mne_raw object) contains the data and channels to investigate
    :param bids_path: (mne_bids object) path to the _channel.tsv file
    :param channels_description: (str or list) contain the channels descriptions to set as bad channels.
    :param subject_info: (subject_info object) contains info about the participants
    :param discard_bads: (boolean) whether or not to discard the channels that were marked as bads as well
    :return:
    """
    if isinstance(channels_description, str):
        channels_description = [channels_description]

    # Loading the channels description file:
    channel_info_file = Path(bids_path.directory,
                             'sub-{}_ses-{}_task-{}_channels.tsv'.format(bids_path.subject, bids_path.session,
                                                                         bids_path.task))
    channel_info = pd.read_csv(channel_info_file, sep="\t")
    # Looping through the passed descriptions:
    bad_channels = []
    for desc in channels_description:
        desc_bad_channels = []
        # Looping through each channel:
        for ind, row in channel_info.iterrows():
            if isinstance(row["status_description"], str):
                if desc in row["status_description"]:
                    bad_channels.append(row["name"])
                    desc_bad_channels.append(row["name"])

    # Discarding the channels that were marked as bad as well
    if discard_bads:
        for ind, row in channel_info.iterrows():
            if row["status"] == "bad":
                bad_channels.append(row["name"])
    # Remove any redundancies:
    bad_channels = list(set(bad_channels))
    # Set these channels as bad:
    raw.info['bads'].extend(bad_channels)

    return raw, bad_channels


def laplace_mapping_validator(mapping, data_channels):
    """
    This function checks the mapping against the channels found in the data. If things don't add up, raise an error
    :param mapping: (dict) contains the mapping between channels to reference and which channels to use to do the
    reference. Format: {ch_name: {"ref_1": ch, "ref_2": ch or None}}
    :param data_channels: (list of string) list of the channels found in the data. This is used to control that all
    the channels found in the mapping are indeed found in the data, to avoid issues down the line
    :return:
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
    The reference mapping in the mapping file is agnostic to which channels are bad, it is only based on the grid
    and strips organization. This function integrates the bad channels information to the mapping. With laplace mapping,
    there are two cases in which a channel should be rejected: if the channel being referenced is bad or if both
    reference channels are bad. This channels identifies these cases and updates the mapping such that those channels
    are excluded. Note that in the case where only one of the two reference is bad but not the other, only the other
    will be used, reducing to a bipolar.
    Additionally, if some of the channels found in the data are not found in the mapping, the data can
    be set as bad if the discard_no_ref_channels is set to True
    :param reference_mapping: (dict) contain for each channel the reference channel according to our scheme
    :param bad_channels: (string list) list of the channels that are marked as bad
    :param all_channels: (string list) list of all the channels
    :return:
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
    Otherwise, you would be referencing  wrong triplets!
    :param to_ref: (numpy array) contains the data to reference.
    :param ref_1: (numpy array) Contains the first ref to do the reference (the ref_1 in mean(ref_1, ref_2)). Dimension
    must match to_ref
    :param ref_2: (numpy array) Contains the second ref to do the reference (the ref_2 in mean(ref_1, ref_2)). Dimension
    must match to_ref
    :return: referenced_data (numpy array) data that were referenced
    """
    # Check that the sizes match:
    if not to_ref.shape == ref_1.shape == ref_2.shape:
        raise Exception("The dimension of the data to subtract do not match the data!")
    referenced_data = to_ref - np.nanmean([ref_1, ref_2], axis=0)

    return referenced_data


def project_elec_to_surf(raw, subjects_dir, subject, montage_space="T1"):
    """
    This function project surface electrodes onto the brain surface to avoid having them floating a little.
    :param raw: (mne raw object)
    :param subjects_dir: (path or string) path to the freesurfer subject directory
    :param subject: (string) name of the subject
    :param montage_space: (string) montage space. If T1, then the projection will be done to the T1 scan of the subject
    If MNI, will be done to fsaverage surface.
    :return:
    """
    # Loading the left and right pial surfaces:
    if montage_space == "T1":
        left_surf = read_geometry(Path(subjects_dir, subject, "surf", "lh.pial"))
        right_surf = read_geometry(Path(subjects_dir, subject, "surf", "rh.pial"))
    elif montage_space == "MNI":
        sample_path = mne.datasets.sample.data_path()
        subjects_dir = Path(sample_path, 'subjects')
        fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # Downloading the data if needed
        subject = "fsaverage"
        left_surf = read_geometry(Path(subjects_dir, subject, "surf", "lh.pial"))
        right_surf = read_geometry(Path(subjects_dir, subject, "surf", "rh.pial"))
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
                          subjects_dir=None, subject=None, montage_space=None):
    """
    This function performs laplacian referencing by subtracting the average of two neighboring electrodes to the
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
    :param raw: (mne raw object) contains the data to reference
    :param reference_mapping: (dict or None) dict of the format described above or None. If None, then the user will
    have the opportunity to create it manually
    :param channel_types: (dict) which channel to consider for the referencing
    :param n_jobs: (int) n_jobs to compute the mapping. Not really useful as we loop through each channel independently
    so setting it to more than 1 will not really do anything. But might be improved in the future.
    :param relocate_edges: (boolean) whether or not to relocate the electrodes that have only one ref!
    :param subjects_dir: (string) directory to the freesurfer data. This is necessary, as the edges get relocated,
    the ecog channels need to be projected to the brain surface.
    :param subject: (string) Name of the subject to access the right surface
    :param montage_space: (string) name of the montage space of the electrodes, either T1 or MNI
    :return:
    mne raw object: with laplace referencing performed.
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
                # Adding the montage back:
                raw.set_montage(montage, on_missing="warn")

    # Projecting the ecog channels to the surface if they were relocated:
    if relocate_edges:
        if len(mne.pick_types(raw.info, ecog=True)) > 0:
            project_elec_to_surf(raw, subjects_dir, subject, montage_space=montage_space)

    return raw, reference_mapping, bad_channels


def remove_channel_tsv_description(channel_tsv_df, description):
    """
    This function removes a specific description from a bids channel tsv file. This is useful if previous iterations
    added wrong annotation to the channels tsv
    :param channel_tsv_df: (df) mne bids channel tsv pandas data frame
    :param description: (string) description to remove
    :return:
    """
    # Find all the channels for which the current description is found
    desc_channels = channel_tsv_df.loc[channel_tsv_df["status_description"].str.contains(description, na=False),
    "name"].to_list()
    for channel in desc_channels:
        # Get the description string of the current channel:
        ch_desc = channel_tsv_df.loc[channel_tsv_df["name"] == channel, "status_description"].item().split("/")
        # Remove the current description:
        ch_desc.remove(description)
        # Add the cleaned string back in:
        channel_tsv_df.loc[channel_tsv_df["name"] == channel, "status_description"] = "/".join(ch_desc)

    return channel_tsv_df


from mne.baseline import rescale


def baseline_scaling(epochs, correction_method="ratio", baseline=(None, 0), picks=None, n_jobs=1):
    """
    This function performs baseline correction on the data. The default is to compute the mean over the entire baseline
    and dividing each data points in the entire epochs by it. Another option is to substract baseline from each time
    point
    :param epochs: (mne epochs object) epochs on which to perform the baseline correction
    :param correction_method: (string) options to do the baseline correction. Options are:
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by
        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')
          source: https://github.com/mne-tools/mne-python/blob/main/mne/baseline.py
    :param baseline: (tuple) which bit to take as the baseline
    :param picks: (None or list of int or list of strings) indices or names of the channels on which to perform the
    correction. If none, all channels are used
    :param n_jobs: (int) number of jobs to use to preprocessing the function. Can be ran in parallel
    :return: none, the data are modified in place
    """
    epochs.apply_function(rescale, times=epochs.times, baseline=baseline, mode=correction_method,
                          picks=picks, n_jobs=n_jobs, )

    return None


def create_mni_montage(channels, bids_path, fs_dir):
    """
    This function fetches the mni coordinates of a set of channels. Importantly, the channels must
    consist of a string with the subject identifier and the channel identifier separated by a minus,
    like: SF102-G1. This ensures that the channels positions can be fecthed from the right subject
    folder. This is specific to iEEG for which we have different channels in each patient which
    may have the same name.
    :param channels: (list) name of the channels for whom to fetch the MNI coordinates. Must contain
    the subject identifier as well as the channel identifier, like SF102-G1.
    :param bids_path: (mne-bids bidsPATH object) contains all the information to fetch the coordinates.
    :param fs_dir: (string or pathlib path object) path to the free surfer root folder containing the fsaverage
    :return: info (mne info object) mne info object with the channels info, including position in MNI space
    """
    from mne_bids import BIDSPath
    # Prepare table to store the coordinates:
    channels_coordinates = []
    # First, extract the name of each subject present in the channels list:
    subjects = list(set([channel.split('-')[0] for channel in channels]))

    for subject in subjects:
        # Extract this participant's channels:
        subject_channels = [channel.split('-')[1] for channel in channels
                            if channel.split('-')[0] == subject]
        # Create the path to this particular subject:
        subject_path = BIDSPath(root=bids_path.root, subject=subject,
                                session=bids_path.session,
                                datatype=bids_path.datatype,
                                task=bids_path.task)
        # Create the name of the mni file coordinates:
        coordinates_file = 'sub-{}_ses-{}_space-fsaverage_electrodes.tsv'.format(subject,
                                                                                 subject_path.session)
        channel_file = 'sub-{}_ses-{}_space-fsaverage_channels.tsv'.format(subject,
                                                                           subject_path.session)
        # Load the coordinates:
        coordinates_df = pd.read_csv(Path(subject_path.directory, coordinates_file), sep='\t')
        channels_df = pd.read_csv(Path(subject_path.directory, coordinates_file), sep='\t')
        # Extract the channels of interest:
        subject_coordinates = coordinates_df.loc[coordinates_df['name'].isin(
            subject_channels), ['name', 'x', 'y', 'z']]
        # Add the channel type:
        subject_coordinates['ch_types'] = [channels_df.loc[channels_df['name'] == channel, 'type']
                                           for channel in subject_coordinates['name']]
        # Make sure to append the name of the subject to the channels coordinates for when we recombine:
        subject_coordinates['name'] = ['-'.join([subject, channel])
                                       for channel in subject_coordinates['name']]
        # Append to the rest
        channels_coordinates.append(subject_coordinates)

    # Concatenating everything into 1 table:
    channels_coordinates = pd.concat(channels_coordinates).reset_index(drop=True)

    # Create one single montage out of it:
    position = channels_coordinates[["x", "y", "z"]].to_numpy()
    # Create the montage:
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(channels, position)),
                                            coord_frame='mni_tal')
    # Making sure to add the mni fiducials:
    montage.add_mni_fiducials(fs_dir)
    # In mne-python, plotting electrodes on the brain requires some additional info about the channels:
    info = mne.create_info(ch_names=channels, ch_types=channels_coordinates['ch_types'].to_list(), sfreq=100)
    # Add the montage:
    info.set_montage(montage)

    return info


def get_roi_channels(channels, rois, bids_path, atlas):
    """
    This function takes in a list of channels and returns only those which are in a particular set of ROIs.
    :param channels: (list of string) list of channels
    :param rois: (list of strings) list of ROIs with names matching the labels of a particular atlas
    :param bids_path: (mne bids path object)
    :param atlas: (string) name of the atlas of interest
    :return: channels (list of string) list of channels found within the region of interest
    """

    # Load the atlas of that particular subject:
    atlas_file = Path(bids_path.root, 'derivatives', 'preprocessing',
                      'sub-' + bids_path.subject, 'ses-' + bids_path.session, bids_path.datatype,
                      'atlas_mapping', 'broadband',
                      'sub-{}_ses-{}_task-{}_desc-atlas_mapping_{}-{}mapping.csv'.format(bids_path.subject,
                                                                                         bids_path.session,
                                                                                         bids_path.task,
                                                                                         bids_path.datatype, atlas))
    atlas_df = pd.read_csv(atlas_file, sep=',')
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

from mne_bids import BIDSPath
bids_root = "C://Users//alexander.lepauvre//Documents//GitHub//iEEG-data-release//bids"
bids_path = BIDSPath(root=bids_root, subject='SF102',
                     session='V1',
                     datatype='ieeg',
                     task='Dur')
subject = 'SF102'
channel = 'LO1'
example_epochs_path = Path(bids_root, 'derivatives', 'preprocessing',
                           'sub-' + subject, 'ses-' + "V1", 'ieeg',
                           "epoching", 'high_gamma',
                           "sub-{}_ses-{}_task-{}_desc-epoching_{}-epo.fif".format(subject,
                                                                                   "V1", "Dur",
                                                                                   "ieeg"))

