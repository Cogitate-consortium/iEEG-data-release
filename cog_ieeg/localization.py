"""
This module provides functions for localization and projection of intracranial EEG (iEEG) electrodes onto brain surfaces.

The functions in this module are designed to handle various aspects of electrode localization, including adding fiducials,
projecting electrodes onto brain surfaces, converting coordinates from MRI to MNI space, and mapping regions of interest (ROIs)
to specific channels.

Author:
-------
Alex Lepauvre
Katarina Bendtz
Simon Henin

License:
--------
This code is licensed under the MIT License.
"""

import math
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import fetch_fsaverage
from mne_bids import convert_montage_to_mri, BIDSPath
from nibabel.freesurfer import read_geometry

from cog_ieeg.utils import path_generator, save_param


def add_fiducials(montage, fs_directory, subject_id):
    """
    Add the estimated fiducials to the montage and compute the transformation.

    Parameters
    ----------
    montage : mne.channels.DigMontage
        Data to which the fiducials should be added.
    fs_directory : str
        Path to the FreeSurfer directory.
    subject_id : str
        Name of the subject.

    Returns
    -------
    mne.channels.DigMontage
        The montage with added fiducials.
    str or mne.transforms.Transform
        The transformation matrix or identifier.
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


def project_elec_to_surf(raw, subjects_dir, subject):
    """
    Project surface electrodes onto the brain surface to avoid having them floating.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw iEEG data.
    subjects_dir : str or pathlib.Path
        Path to the FreeSurfer subject directory.
    subject : str
        Name of the subject.

    Returns
    -------
    mne.io.Raw
        The raw data with electrodes projected onto the brain surface.
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


def project_montage_to_surf(montage, channel_types, subject, fs_dir):
    """
    Project the electrodes to the pial surface. Note that this is only ever done on the ECoG channels.

    Parameters
    ----------
    montage : mne.channels.DigMontage
        The montage to project.
    channel_types : dict
        Dictionary specifying channel types.
    subject : str
        Name of the subject.
    fs_dir : str or pathlib.Path
        Path to the FreeSurfer directory.

    Returns
    -------
    mne.channels.DigMontage
        The updated montage with electrodes projected onto the brain surface.
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
    Fetch the MNI coordinates of a set of channels from the BIDS directory.

    Parameters
    ----------
    channels : list of str
        List of channel names to fetch MNI coordinates for.
    bids_path : mne_bids.BIDSPath
        BIDSPath object with information to fetch coordinates.
    fsaverage_dir : str
        Path to the FreeSurfer root folder containing the fsaverage.

    Returns
    -------
    mne.Info
        Info object with channel info, including positions in MNI space.
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
    Get channels that are in a particular set of ROIs.

    Parameters
    ----------
    channels : list of str
        List of channels.
    rois : list of str
        List of ROIs with names matching the labels of a particular atlas.
    bids_path : mne_bids.BIDSPath
        BIDSPath object pointing to the subject's BIDS directory.
    atlas : str
        Name of the atlas of interest.

    Returns
    -------
    list of str
        List of channels found within the region of interest.
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
            print(f'WARNING: No loc for {channel}')
            continue
        except IndexError:
            print(f'WARNING: {channel} not found in channels localization file')
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

    Parameters
    ----------
    montage : mne.channels.DigMontage
        The montage containing electrode positions in MRI coordinates.
    subject : str
        Subject ID.
    fs_dir : str or pathlib.Path
        Path to the FreeSurfer directory.

    Returns
    -------
    dict
        Dictionary with MNI positions of electrodes.
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


def roi_mapping(mne_object, list_parcellations, subject, fs_dir, param, save_root, step, signal, file_prefix,
                file_extension='mapping.csv'):
    """
    Map the electrodes on different atlases.

    Parameters
    ----------
    mne_object : mne.io.Raw or mne.Epochs
        Object containing the montage to extract the ROI.
    list_parcellations : list of str
        List of the parcellation files to use for the mapping.
    subject : str
        Name of the subject to access the FreeSurfer recon.
    fs_dir : str or pathlib.Path
        Path to the FreeSurfer directory containing all the participants.
    param : dict
        Parameters used to generate the data.
    save_root : str or pathlib.Path
        Path to where the data should be saved.
    step : str
        Name of the preprocessing step.
    signal : str
        Name of the signal being saved.
    file_prefix : str
        Prefix of the file to save.
    file_extension : str, optional
        File name extension (default is 'mapping.csv').

    Returns
    -------
    dict of pandas.DataFrame
        One DataFrame per parcellation with the mapping between ROI and channels.
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
            # Convert the montage from mni to mri:
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
    # Creating the directory if it doesn't exist:
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


def exclude_distant_channels(montage, subject, fs_dir, max_dist=5):
    """
    Exclude electrodes that are further than max_dist from the brain surface.

    Parameters
    ----------
    montage : mne.channels.DigMontage
        The montage containing electrode positions.
    subject : str
        Subject ID.
    fs_dir : str or pathlib.Path
        Path to the FreeSurfer directory.
    max_dist : float, optional
        Maximum distance from the brain surface (default is 5 mm).

    Returns
    -------
    mne.channels.DigMontage
        The updated montage with distant electrodes excluded.
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
