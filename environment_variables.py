import os
from pathlib import Path

# Parameters to adjust: 
bids_root = Path(Path(__file__).resolve().parent, "bids")
fs_directory = Path(bids_root, 'derivatives', 'fs') # Change only if you have your freesurfer root elsewhere

# ===============================================================
# Don't touch
# xnat parameters:
xnat_host = 'cogitate-data'
xnat_project = 'COG_IEEG_EXP1_BIDS'
