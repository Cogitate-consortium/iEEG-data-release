import mne
from pathlib import Path

import mne_bids
import numpy as np
from mne.viz import plot_alignment
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt

mne.set_log_level(verbose="ERROR")


def ieeg_bids_test(bids_root_ref, bids_root_new, subject_name_ref, subject_name_new, session, data_type, task,
                   interactive=True):
    """
    This function compares the data after the new implementation of the BIDS conversion. It compares it against the BIDS
    converted data used in the COGITATE, to make sure no issues were introduced during the conversion.
    :param bids_root_ref: (string or path object)
    :param bids_root_new: (string or path object)
    :param subject_name_ref: (string) subject name in the ref directory
    :param subject_name_new: (string) subject name in the new directory
    :param session: (string) name of the session to test
    :param data_type: (string) data type
    :param task: (string) name of the task
    :param interactive: (bool) whether the script is ran interactively or not. if yes, the montages vizualization will
    be displayed. If not, it will be skipped.
    """
    print("-" * 40)
    print("Welcome to ieeg_bids_test!")
    print("We will test the data of:")
    print("sub-" + subject_name_new)
    print("session-" + session)
    print("task-" + task)
    print("from: " + bids_root_new)
    print("against the data in: " + bids_root_ref)
    print("-" * 40)

    # ==================================================================================================================
    # Load the data:
    # Ref data:
    bids_path = BIDSPath(root=bids_root_ref, subject=subject_name_ref,
                         session=session,
                         datatype=data_type,
                         task=task)
    raw_ref = read_raw_bids(bids_path=bids_path)
    raw_ref.load_data()
    # New data:
    bids_path = BIDSPath(root=bids_root_new, subject=subject_name_new,
                         session=session,
                         datatype=data_type,
                         task=task)
    raw_new = read_raw_bids(bids_path=bids_path)
    raw_new.load_data()

    # ==================================================================================================================
    # Compare the ephys data:
    # Extract data matrices:
    data_ref = raw_ref.get_data()
    data_new = raw_new.get_data()

    # Check that the dimensions agree:
    assert np.all(raw_ref.times == raw_ref.times), "The time vector differs between both versions!"
    assert data_ref.shape == data_new.shape, "The dimensions have changed with the new BIDS conversion!"
    assert np.allclose(data_ref, data_new), "The data themselves are different in the new BIDS conversion!"

    # ==================================================================================================================
    # Compare channels localizations:
    # Compare the montages:
    ref_montage = raw_ref.get_montage()
    new_montage = raw_new.get_montage()
    # Loop through each channel to ensure that everything is the same:
    for ch in raw_ref.ch_names:
        # Find the indices of that channel in each:
        new_ind = [ind for ind, new_ch in enumerate(new_montage.ch_names) if new_ch == ch][0]
        ref_ind = [ind for ind, ref_ch in enumerate(ref_montage.ch_names) if ref_ch == ch][0]
        assert np.allclose(ref_montage.dig[ref_ind + 3]['r'], new_montage.dig[new_ind]['r']), (
            "The coordinates are not equal for {}".format(ch))

    # Finally, checking the annotations:
    assert np.all(raw_ref.annotations.onset == raw_ref.annotations.onset), \
        "The annotations time stamps does not match!!"
    assert np.all(raw_ref.annotations.duration == raw_ref.annotations.duration), \
        "The annotations duations does not match!!"
    assert np.all(raw_ref.annotations.description == raw_ref.annotations.description), \
        "The annotations descriptions does not match!!"

    if interactive:
        # Vizualize both montages:
        # Respecify the reference montage to be in MRI space:
        ref_montage = raw_ref.get_montage()
        positions = ref_montage.get_positions()['ch_pos']
        ref_montage = mne.channels.make_dig_montage(ch_pos=positions, coord_frame="mri")
        # Add estimated fiducialts and computer transform:
        ref_montage.add_estimated_fiducials("sub-" + subject_name_ref,
                                            subjects_dir=Path(bids_root_ref, "derivatives", "fs"))
        trans = mne.channels.compute_native_head_t(ref_montage)
        # Add to the data:
        raw_ref.set_montage(ref_montage, on_missing="warn")
        # Plot the montage:
        plot_alignment(
            raw_ref.info,
            trans=trans,
            subject="sub-" + subject_name_ref,
            subjects_dir=Path(bids_root_ref, "derivatives", "fs"),
            surfaces={"pial": 0.5},
            coord_frame="mri",
            sensor_colors=(1.0, 1.0, 1.0, 0.5),
        )
        plt.subplots(figsize=(10, 10))
        plt.show()

        # In comparison plot the new montage:
        new_montage = raw_new.get_montage()
        # Convert the montage to MRI:
        mne_bids.convert_montage_to_mri(new_montage, "sub-" + subject_name_new,
                                        subjects_dir=Path(bids_root_new, "derivatives", "fs"))
        new_montage.add_estimated_fiducials("sub-" + subject_name_new,
                                            subjects_dir=Path(bids_root_new, "derivatives", "fs"))
        trans = mne.channels.compute_native_head_t(new_montage)
        raw_new.set_montage(new_montage, on_missing="warn")
        plot_alignment(
            raw_new.info,
            trans=trans,
            subject="sub-" + subject_name_new,
            subjects_dir=Path(bids_root_new, "derivatives", "fs"),
            surfaces={"pial": 0.5},
            coord_frame="head",
            sensor_colors=(1.0, 1.0, 1.0, 0.5),
        )
        plt.subplots(figsize=(10, 10))
        plt.show()

    print("-" * 40)
    print("Congrats, everything is the same, nothing went wrong in the conversion!")


if __name__ == "__main__":
    ieeg_bids_test(r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids",
                   r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\data_release\bids",
                   "SE103", "CE103", "V1", "ieeg", "Dur",
                   interactive=True)
