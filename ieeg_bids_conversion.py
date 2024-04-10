import mne
import shutil
import os
import pandas as pd
from pathlib import Path
from mne_bids import BIDSPath, write_raw_bids, convert_montage_to_ras
import environment_variables as ev
mne.set_log_level(verbose="ERROR")


def ieeg_bids_converter(subject_id, session, task):
    """
    This functions converts the raw data from the COGITATE to BIDS standard format.
    :param subject_id: (string) subject identifier
    :param session: (string) session identifier
    :param task: (string) task name
    :return: None
    """
    print("-" * 40)
    print("Welcome to ieeg_bids_converter!")
    print("We will convert the data of:")
    print("sub-" + subject_id)
    print("session-" + session)
    print("task-" + task)
    print("The data will be saved under: " + ev.bids_root)
    # @Praveen, here is where you will need to adjust the path to the files and folders namings based on the release
    # naming conventions!
    events_tsv_file = Path(ev.raw_root, subject_id, subject_id + "_ECOG_1",
                           "RESOURCES", "sub-{}_ses-{}_task-{}_events.tsv".format(subject_id.replace('C', 'S'),
                                                                                  session, task))
    electrodes_tsv_file = Path(ev.raw_root, subject_id, subject_id + "_ECOG_1",
                               "RESOURCES",
                               "sub-{}_ses-{}_space-Other_electrodes.tsv".format(subject_id.replace('C', 'S'),
                                                                                 session))
    channels_tsv_file = Path(ev.raw_root, subject_id, subject_id + "_ECOG_1",
                             "RESOURCES",
                             "sub-{}_ses-{}_task-{}_channels.tsv".format(subject_id.replace('C', 'S'),
                                                                         session, task))
    laplace_mapping_src = Path(ev.raw_root, subject_id, subject_id + "_ECOG_1", "RESOURCES",
                               "sub-{}_ses-{}_laplace_mapping_ieeg.json".format(subject_id.replace('C', 'S'),
                                                                                session))
    fs_recon_src = Path(r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids_old\derivatives\fs",
                        "sub-" + subject_id.replace('C', 'S'))

    # ==================================================================================================================
    # Load data:
    # 1. iEEG data:
    # Find files:
    raw_data_path = Path(ev.raw_root, subject_id, subject_id + "_ECOG_1",
                         "scans", "EDF").glob('*_{}*.EDF'.format(task))
    raw_data_files = [x for x in raw_data_path if x.is_file()]
    # Load each of them:
    raws_list = []
    for raw_file in raw_data_files:
        raw_i = mne.io.read_raw_edf(raw_file, preload=True)
        raws_list.append(raw_i)
    # Concatenate all the files:
    raw = mne.concatenate_raws(raws_list)
    # 2. Additional data:
    # Load the additional files:
    events_tsv = pd.read_csv(events_tsv_file, sep='\t')
    electrodes_tsv_t1 = pd.read_csv(electrodes_tsv_file, sep='\t')
    channels_tsv = pd.read_csv(channels_tsv_file, sep='\t')

    # ==================================================================================================================
    # Combine iEEG data and metadata:
    # 1. Convert the events tsv to annotations:
    exp_annotations = mne.Annotations(events_tsv["onset"].to_numpy(),
                                      events_tsv["duration"].to_numpy(),
                                      events_tsv["trial_type"].to_numpy(),
                                      orig_time=raw.annotations.orig_time)
    # Combine with existing annotations:
    all_annotations = raw.annotations.__add__(exp_annotations)
    raw.set_annotations(all_annotations)

    # 2. Set the channels types:
    channel_types_mapping = dict(zip(channels_tsv["name"].to_list(),
                                     channels_tsv["type"].str.lower().to_list()))
    # Extract channel types from channels.tsv
    raw.set_channel_types(channel_types_mapping)  # Add to the raw
    # Set the misc channels to bads:
    misc_channels = [ch for ch in raw.ch_names if ch not in channels_tsv["name"].to_list()]
    raw.drop_channels(misc_channels)

    # 3. Create the T1 montage:
    channels = electrodes_tsv_t1["name"].tolist()  # Extract channels names
    position = electrodes_tsv_t1[["x", "y", "z"]].to_numpy()  # Extract channels positions
    montage_t1 = mne.channels.make_dig_montage(ch_pos=dict(zip(channels, position)),
                                               coord_frame="mri")  # Create T1 dig montage

    # Create the BIDS path
    bids_path = BIDSPath(root=ev.bids_root, subject=subject_id, session=session, task=task, space="Other",
                         datatype="ieeg")

    # ==================================================================================================================
    # Save the data to BIDS:
    write_raw_bids(
        raw,
        bids_path,
        montage=montage_t1,
        acpc_aligned=False,
        overwrite=True,
        allow_preload=True,
        format='BrainVision'
    )

    # In addition, fetch the Laplace montage file:
    laplace_mapping_dst = Path(bids_path.directory, 'ieeg',
                               "sub-{}_ses-{}_laplace_mapping_ieeg.json".format(subject_id, session))
    # Copy to the bids directory:
    shutil.copyfile(laplace_mapping_src, laplace_mapping_dst)

    # Copy paste the free surfer directory to the bids derivatives:
    fs_root = ev.fs_directory
    if not os.path.isdir(fs_root):
        os.makedirs(fs_root)

    # Copy the whole folder in this directory:
    fs_recon_dst = Path(fs_root, "sub-" + subject_id)
    try:
        shutil.copytree(fs_recon_src, fs_recon_dst)
    except FileExistsError:
        print("WARNING: The free surfer data for this subject were already saved in the specified directory")

    # Finally, write the original channel tsv over the one generated by MNE, as it contains extra channel descriptions:
    channels_tsv.to_csv(Path(bids_path.directory, 'ieeg',
                             "sub-{}_ses-{}_task-{}_channels.tsv".format(subject_id, session, task)),
                        sep='\t', index=False)


if __name__ == "__main__":
    ieeg_bids_converter("CF102", "V1", "Dur")
