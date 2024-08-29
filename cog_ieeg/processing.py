"""
This module provides functions for processing intracranial EEG (iEEG) data, including preprocessing steps
and signal processing techniques.

The functions in this module handle various aspects of iEEG data processing, such as detrending, filtering,
baseline correction, and common average referencing (CAR).

Author:
-------
Alex Lepauvre
Katarina Bendtz
Simon Henin

License:
--------
This code is licensed under the MIT License.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from cog_ieeg.localization import project_elec_to_surf


def detrend_runs(raw, njobs=1):
    """
    Detrend the run. If the raw data initially consisted of several files that were concatenated,
    the detrending will be performed separately for each run. Otherwise, it will be done in one go.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data.
    njobs : int, optional
        Number of jobs to use for detrending, by default 1.

    Returns
    -------
    mne.io.Raw
        Detrended raw iEEG data.
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


def custom_car(raw, reference_channel_types=None, target_channel_types=None):
    """
    Compute a custom common average reference (CAR) by averaging the amplitude across specific channel types and
    subtracting it from all channels at each time point.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data to be rereferenced.
    reference_channel_types : dict, optional
        Dictionary specifying the channel types to take as reference, by default None.
    target_channel_types : dict, optional
        Dictionary specifying the channel types to apply the reference to, by default None.

    Returns
    -------
    mne.io.Raw
        Modified raw iEEG data with CAR applied.
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


def epoching(raw, events, events_dict, picks="all", tmin=-0.5, tmax=2.0, events_not_to_epoch=None,
             baseline=(None, 0.0), reject_by_annotation=True, meta_data_column=None):
    """
    Perform epoching according to the provided parameters.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data to epoch.
    events : np.ndarray
        MNE events array, three columns: event time stamp, event ID.
    events_dict : dict
        Dictionary mapping event IDs to their descriptions.
    picks : str or list of int or list of str, optional
        Channels to epoch, by default "all".
    tmin : float, optional
        How long before each event of interest to epoch (in seconds), by default -0.5.
    tmax : float, optional
        How long after each event of interest to epoch (in seconds), by default 2.0.
    events_not_to_epoch : list of str, optional
        Names of the events not to epoch, by default None.
    baseline : tuple of float, optional
        Tuple passed to the MNE epoching function to apply baseline correction, by default (None, 0.0).
    reject_by_annotation : bool, optional
        Whether to discard trials based on annotations, by default True.
    meta_data_column : list of str, optional
        List of column names for the metadata, by default None.

    Returns
    -------
    mne.Epochs
        The epoched iEEG data.
    """
    if picks == "all":
        picks = raw.info["ch_names"]
    if isinstance(baseline, list):
        baseline = tuple(baseline)
    print('Performing epoching')

    # We only want to epoch certain events. The config file specifies which ones should not be epoched (fixation
    # for ex). The function below returns only the events we are interested in!
    if events_not_to_epoch is not None:
        events_of_interest = {key: events_dict[key] for key in events_dict if not
        any(substring in key for substring in events_not_to_epoch)}
    else:
        events_of_interest = events_dict

    # Epoching the data:
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
    Compute the high gamma signal by filtering the signal in bins between a defined frequency range.
    In each frequency bin, the Hilbert transform is applied and the amplitude is extracted (i.e., envelope).
    The envelope is then normalized by dividing by the mean amplitude over the entire time series (within each channel separately).
    The amplitude is then averaged across frequency bins.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data.
    frequency_range : list of list of float, optional
        Frequency range of interest, by default None.
    bands_width : float or int, optional
        Width of the frequency bands to loop over, by default 10.
    njobs : int, optional
        Number of parallel processes to compute the high gammas, by default 1.
    channel_types : dict, optional
        Dictionary specifying the channel types for which to compute the high gamma, by default None.
    do_baseline_normalization : bool, optional
        Whether to perform baseline normalization, by default True.

    Returns
    -------
    mne.io.Raw
        The raw object containing the high gamma signal.
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

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data.
    frequency_range : list of list of float, optional
        Frequency range of interest, by default None.
    njobs : int, optional
        Number of parallel processes to compute the ERP, by default 1.
    channel_types : dict, optional
        Dictionary specifying the channel types for which to compute the ERP, by default None.
    kwargs : dict
        Additional arguments passed to the MNE raw.filter function.

    Returns
    -------
    mne.io.Raw
        The raw object containing the computed ERP signal.
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


def description_ch_rejection(raw, bids_path, channels_description, discard_bads=True):
    """
    Discard channels based on the descriptions found in the _channel.tsv file in the BIDS dataset.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data containing the channels to investigate.
    bids_path : mne_bids.BIDSPath
        BIDSPath object pointing to the _channel.tsv file.
    channels_description : str or list of str
        Channels descriptions to set as bad channels.
    discard_bads : bool, optional
        Whether to discard the channels that were marked as bad, by default True.

    Returns
    -------
    tuple
        A tuple containing the updated raw object and a list of bad channels.
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

    Parameters
    ----------
    mapping : dict
        Contains the mapping between channels to reference and the channels to use for the reference.
    data_channels : list of str
        List of the channels found in the data.

    Returns
    -------
    None
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
    # Checking that there is never the case of having both references as None:
    if any([mapping[channel]["ref_1"] is None and mapping[channel]["ref_2"] is None for channel in mapping.keys()]):
        invalid_channels = [channel for channel in mapping.keys() if
                            mapping[channel]["ref_1"] is None and mapping[channel]["ref_2"] is None]
        mne.utils.warn("The channels {0} have two None references. They will be set to bad! If this is not intended,"
                       "please review your mapping!".format(invalid_channels))

    return None


def remove_bad_references(reference_mapping, bad_channels, all_channels):
    """
    Integrate bad channels information into the reference mapping. Update the mapping such that channels with bad
    references are excluded.

    Parameters
    ----------
    reference_mapping : dict
        Dictionary containing the reference channel mapping.
    bad_channels : list of str
        List of channels that are marked as bad.
    all_channels : list of str
        List of all channels.

    Returns
    -------
    tuple
        A tuple containing the updated reference mapping and the updated list of bad channels.
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
        # But if only one of the two references is bad, setting that one ref to None:
        elif reference_mapping[channel]["ref_1"] in bad_channels:
            new_reference_mapping[channel]["ref_1"] = None
        elif reference_mapping[channel]["ref_2"] in bad_channels:
            new_reference_mapping[channel]["ref_2"] = None

        # As a result of setting one of the references to None if bad, some channels located close to edges might have
        # only None as references, in which case they can't be referenced. These channels need to be removed from the
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
    Compute the Laplace reference by subtracting the mean of ref_1 and ref_2 from the channel to reference:
    ch = ch - mean(ref_1, ref_2).

    Parameters
    ----------
    to_ref : np.ndarray
        Array containing the data to reference.
    ref_1 : np.ndarray, optional
        Array containing the first reference channel data, by default None.
    ref_2 : np.ndarray, optional
        Array containing the second reference channel data, by default None.

    Returns
    -------
    np.ndarray
        The referenced data.
    """
    # Check that the sizes match:
    if not to_ref.shape == ref_1.shape == ref_2.shape:
        raise Exception("The dimension of the data to subtract do not match the data!")
    referenced_data = to_ref - np.nanmean([ref_1, ref_2], axis=0)

    return referenced_data


def laplacian_referencing(raw, reference_mapping, channel_types=None,
                          n_jobs=1, relocate_edges=True,
                          subjects_dir=None, subject=None):
    """
    Perform Laplacian referencing by subtracting the average of two neighboring electrodes from the
    central one.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data to reference.
    reference_mapping : dict or None
        Dictionary of the format described above or None.
    channel_types : dict, optional
        Dictionary specifying which channels to consider for referencing, by default None.
    n_jobs : int, optional
        Number of jobs to compute the referencing, by default 1.
    relocate_edges : bool, optional
        Whether to relocate the electrodes that have only one reference, by default True.
    subjects_dir : str, optional
        Directory to the FreeSurfer data, by default None.
    subject : str, optional
        Name of the subject to access the right surface, by default None.

    Returns
    -------
    tuple
        A tuple containing the modified raw object, the updated reference mapping, and the updated list of bad channels.
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
    # Performing the Laplacian reference:
    # Extract data to get the references and avoid issue with changing in place when looping:
    ref_data = raw.get_data()
    data_chs = raw.ch_names
    # Get the size of a channel matrix to handle the absence of ref_2 for corners:
    mat_shape = np.squeeze(raw.get_data(picks=0)).shape
    empty_mat = np.empty(mat_shape)
    empty_mat[:] = np.nan
    # Extract the montage:
    montage = raw.get_montage()
    # performing the Laplacian referencing:
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
            # If one of the two references is only NaN, then there was one reference missing, in which case the channel must
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

    # Projecting the ECoG channels to the surface if they were relocated:
    if relocate_edges:
        ecog_channels = mne.pick_types(raw.info, ecog=True)
        if len(ecog_channels) > 0:
            project_elec_to_surf(raw, subjects_dir, subject)

    return raw, reference_mapping, bad_channels


def baseline_scaling(epochs, correction_method="ratio", baseline=(None, 0), picks=None, n_jobs=1):
    """
    Perform baseline correction on the data.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs on which to perform the baseline correction.
    correction_method : str, optional
        Method to use for the baseline correction, by default "ratio".
    baseline : tuple, optional
        Time interval to use as the baseline, by default (None, 0).
    picks : list of int or list of str, optional
        Indices or names of the channels on which to perform the correction, by default None.
    n_jobs : int, optional
        Number of jobs to use to preprocess the function, by default 1.

    Returns
    -------
    None
    """
    from mne.baseline import rescale
    epochs.apply_function(rescale, times=epochs.times, baseline=baseline, mode=correction_method,
                          picks=picks, n_jobs=n_jobs, )

    return None


def notch_filtering(raw, njobs=1, frequency=60, remove_harmonics=True, filter_type="fir",
                    cutoff_lowpass_bw=None, cutoff_highpass_bw=None, channel_types=None):
    """
    Filter the raw data according to the set parameters.

    Parameters
    ----------
    raw : mne.io.Raw
        The continuous raw iEEG data.
    njobs : int, optional
        Number of jobs to preprocess the filtering in parallel threads, by default 1.
    frequency : int or float, optional
        Frequency to notch out, by default 60.
    remove_harmonics : bool, optional
        Whether to remove all the harmonics of the declared frequency, by default True.
    filter_type : str, optional
        Type of filter to use, "iir" or "fir", by default "fir".
    cutoff_lowpass_bw : float, optional
        Frequency for low pass (only used if type is "iir"), by default None.
    cutoff_highpass_bw : float, optional
        Frequency for high pass (only used if type is "iir"), by default None.
    channel_types : dict, optional
        Dictionary specifying the channel types to notch filter, by default None.

    Returns
    -------
    mne.io.Raw
        The notch filtered raw data.
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
    Parse the events found in the epochs descriptions to create the metadata.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs for which the metadata will be generated.
    metadata_column_names : list of str
        Names of the columns of the metadata.

    Returns
    -------
    mne.Epochs
        The epochs with added metadata.
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
