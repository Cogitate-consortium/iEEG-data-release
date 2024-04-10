import mne
from mne.viz import plot_alignment, snapshot_brain_montage
import pandas as pd
from pathlib import Path
from mne_bids import convert_montage_to_mri
import matplotlib.pyplot as plt
from mne_bids import BIDSPath, write_raw_bids


# Roots
bids_root_old = r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids_old"
bids_root_new = r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids_new"
# Locs:
locs_old = pd.read_csv(Path(bids_root_old, "sub-SF102", "ses-1", "ieeg",
                            "sub-SF102_ses-1_space-Other_electrodes.tsv"),
                       sep='\t')
# Raws:
raw_old = mne.io.read_raw_brainvision(Path(bids_root_old, "sub-SF102", "ses-1", "ieeg",
                                           "sub-SF102_ses-V1_task-Dur_ieeg.vhdr"))
channel_types_mapping = dict(zip(raw_old.info["ch_names"],
                                 ["ecog" for i in range(len(raw_old.info["ch_names"]))]))
# Extract channel types from channels.tsv
raw_old.set_channel_types(channel_types_mapping)  # Add to the raw

channels = locs_old["name"].tolist()  # Extract channels names
position = locs_old[["x", "y", "z"]].to_numpy()  # Extract channels positions

montage_t1 = mne.channels.make_dig_montage(ch_pos=dict(zip(channels, position)),
                                           coord_frame="mri")  # Create T1 dig montage

bids_path = BIDSPath(root=bids_root_new, subject="SF102", session="V1", task="Dur", space="Other", datatype="ieeg")

write_raw_bids(
    raw_old,
    bids_path,
    montage=montage_t1,
    acpc_aligned=False,
    overwrite=True,
    allow_preload=True,
    format='BrainVision'
)

