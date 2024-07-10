from pathlib import Path

# Parameters to adjust: 
bids_root = r"C:\Users\alexander.lepauvre\Documents\GitHub\iEEG-data-release\bids-xnat"
fs_directory = Path(bids_root, 'derivatives', 'fs') # Change only if you have your freesurfer root elsewhere

# ===============================================================
# Don't touch
# xnat parameters:
xnat_host = 'xnat-curate'
xnat_project = 'COG_IEEG_EXP1_BIDS_SAMPLE'
