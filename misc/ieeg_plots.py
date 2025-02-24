# Loading all modules:
import os.path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import mne
from mne_bids import BIDSPath, read_raw_bids

from cog_ieeg.vizualization import count_colors
from cog_ieeg.utils import (get_bids_root, get_fs_directory, set_bids_root, get_pipeline_config, 
                            create_default_config, set_xnat_host, set_xnat_project, print_config)

def inch2cm(cm):
    return cm * 2.54

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
dpi = 300
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Fetch fsaverage:
mne.datasets.fetch_fsaverage(subjects_dir=get_fs_directory(), verbose=None)

# Specify the subjects to work with:
subjects_df = pd.read_csv(Path(get_bids_root(), "participants.tsv"), sep='\t')
subjects = subjects_df["participant_id"].to_list()

# Create a histogram of age distribution:
import pylustrator
pylustrator.start()
fig, ax = plt.subplots(4, 1, figsize=(4, 16))
ax[0].pie(subjects_df['sex'].value_counts(), labels=subjects_df['sex'].value_counts().index,
          colors=[[246 / 255, 216 / 255, 174 / 255], [34 / 255, 124 / 255, 157 / 255]])
ax[0].set_title("Sex")
ax[1].pie(subjects_df['handedness'].value_counts(), labels=subjects_df['handedness'].value_counts().index,
          colors=[[246 / 255, 216 / 255, 174 / 255], [34 / 255, 124 / 255, 157 / 255], [85 / 255, 0 / 255, 0 / 255]])
ax[1].set_title("handedness")
ax[2].pie(subjects_df['primary_language'].value_counts(), labels=subjects_df['primary_language'].value_counts().index,
          colors=[[246 / 255, 216 / 255, 174 / 255], [34 / 255, 124 / 255, 157 / 255], [85 / 255, 0 / 255, 0 / 255]])
ax[2].set_title("Primary Language")
ax[3].hist(subjects_df['age'])
ax[3].set_title("Age")
ax[3].set_ylabel('# subjects')
ax[3].set_xlabel('Age (yrs.)')
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(10.090000/2.54, 27.740000/2.54, forward=True)
plt.figure(1).axes[0].set(position=[0.1566, 0.812, 0.6743, 0.1674])
plt.figure(1).axes[0].set_position([0.182178, 0.750627, 0.609931, 0.222078])
plt.figure(1).axes[0].texts[0].set(position=(-0.1309, 0.4486), weight='bold')
plt.figure(1).axes[0].texts[1].set(position=(0.0961, -0.4486), color='#ffffffff', weight='bold')
plt.figure(1).axes[1].set(position=[0.1566, 0.6384, 0.6743, 0.1674])
plt.figure(1).axes[1].set_position([0.182178, 0.520234, 0.609931, 0.222078])
plt.figure(1).axes[1].texts[0].set(position=(-0.1309, 0.4887), weight='bold')
plt.figure(1).axes[1].texts[1].set(position=(0.4677, -0.4319), color='#ffffffff', weight='bold')
plt.figure(1).axes[1].texts[2].set(position=(0.2762, 0.1149), weight='bold')
plt.figure(1).axes[2].set(position=[0.1566, 0.4414, 0.6743, 0.1674])
plt.figure(1).axes[2].set_position([0.182178, 0.258914, 0.609931, 0.222078])
plt.figure(1).axes[2].texts[0].set(position=(0.0961, 0.5329), weight='bold')
plt.figure(1).axes[2].texts[1].set(position=(0.6498, -0.4925), weight='bold')
plt.figure(1).axes[2].texts[2].set(position=(0.2762, 0.1074), text='Spanish/English', weight='bold')
plt.figure(1).axes[3].set(position=[0.227, 0.3105, 0.5262, 0.1161])
plt.figure(1).axes[3].set_position([0.223979, 0.085197, 0.519164, 0.154042])
plt.show()
#% end: automatic generated code from pylustrator
save_path = Path("misc", "coverage_plots")
if not os.path.isdir(save_path):
    os.makedirs(save_path)
plt.savefig(Path(save_path, "demographics.svg"))
plt.close()



# Further specify the data to use from the subjects:
session = "1"
datatype = "ieeg"
task = "Dur"
atlas = "destrieux"


# Pre allocate
montages = []
subjects_implant_type = []
lateralization = []
mni_coords = {}
channels_types_mapping = {}
labels_df = []

# Loop through each subject:
for i, subject in enumerate(subjects):
    print(subject)
    # ===================================================================
    # Load the data:
    # Set the bids path:
    bids_path = BIDSPath(root=get_bids_root(), subject=subject.split("-")[1],
                         session=session,
                         datatype=datatype,
                         task=task)
    # Load the data
    raw = read_raw_bids(bids_path=bids_path, verbose=True)

    # ===================================================================
    # Handle electrodes:

    # Extract the montage:
    montage = raw.get_montage()
    mni_montage = montage.get_positions()['ch_pos']
    mni_montage = {"-".join([subject, key]): val for key, val in mni_montage.items()}
    mni_coords.update(mni_montage)

    # Load the anatomical labels:
    labels_file = Path(bids_path.directory, 'sub-{}_ses-{}_atlas-{}_labels.tsv'.format(subject.split("-")[1],
                       session, atlas))
    subject_labels = pd.read_csv(labels_file, sep='\t')
    labels_df.append(subject_labels)

    # =================================
    # Get the lateralization
    positions = montage.get_positions()
    x_pos = np.array([positions['ch_pos'][ch][0] for ch in positions['ch_pos'].keys()
                      if not np.isnan(positions['ch_pos'][ch][0])])
    if np.all(x_pos > 0):
        lateralization.append("right")
    elif np.all(x_pos < 0):
        lateralization.append("left")
    else:
        lateralization.append("bilateral")
    # Extract the channel types for this subject:
    implant_type = [ch_type for ch_type in list(set(raw.get_channel_types())) if ch_type in ['ecog', 'seeg']]
    subjects_implant_type.append(implant_type)

    # =================================
    # Channel types mapping:
    ch_type_map = dict(zip(raw.info.ch_names, raw.get_channel_types()))
    # Keep only the ecog and seeg channels:
    channels_types_mapping.update({"-".join([subject, ch]): typ for ch, typ in ch_type_map.items()
                                   if typ in ['ecog', 'seeg']})

# ==================================================================================
# Summary plots:
# Pie chart of channels counts:
# Count how many of each type:
types_counts = Counter(list(channels_types_mapping.values()))
fig, ax = plt.subplots(3, 1, figsize=(4, 12))
ax = ax.flatten()
ax[0].pie(list(types_counts.values()), labels=list(types_counts.keys()),
          autopct=lambda x: '{:.0f}'.format(x * np.sum(list(types_counts.values())) / 100),
          colors=[[246 / 255, 216 / 255, 174 / 255], [34 / 255, 124 / 255, 157 / 255]])
ax[0].set_title("Channels")
# Venn diagram of subjects with ecog, seeg and both:
# Initialize counts
ecog_count = 0
seeg_count = 0
both_count = 0

# Count occurrences
for t in subjects_implant_type:
    if 'ecog' in t and 'seeg' in t:
        both_count += 1
    elif 'ecog' in t:
        ecog_count += 1
    elif 'seeg' in t:
        seeg_count += 1
venn2(subsets=(ecog_count, seeg_count, both_count), set_labels=('ecog', 'seeg'),
      set_colors=[[246 / 255, 216 / 255, 174 / 255], [34 / 255, 124 / 255, 157 / 255]], ax=ax[1])
ax[1].set_title("Implants")

# Venn diagram of channels lateralization:
left_count = lateralization.count("left") + lateralization.count("bilateral")  # 'left' + 'both'
right_count = lateralization.count("right") + lateralization.count("bilateral")  # 'right' + 'both'
both_count = lateralization.count("bilateral")  # Only 'both'
venn_counts = (left_count - both_count, right_count - both_count, both_count)
# Create the Venn diagram
venn2(subsets=venn_counts, set_labels=('Left Implant', 'Right Implant'),
      set_colors=[[160 / 255, 206 / 255, 217 / 255], [255 / 255, 191 / 255, 70 / 255]], ax=ax[2])
ax[2].set_title("Scheme")
plt.savefig(Path(save_path, "channels_summaries.svg"))
plt.close()

# ==================================================================================
# Single electrodes plots:
# Create the mni montage:
mni_montage = mne.channels.make_dig_montage(ch_pos=mni_coords,
                                            coord_frame='mni_tal')
# Project the montage to the surface:
mni_montage = project_montage_to_surf(mni_montage, channels_types_mapping, "fsaverage", get_fs_directory())
# Remove electrodes that are too far away:
mni_montage = exclude_distant_channels(mni_montage, "fsaverage", get_fs_directory(), max_dist=3)
# Add the MNI fiducials
mni_montage.add_mni_fiducials(get_fs_directory())
# In mne-python, plotting electrodes on the brain requires some additional info about the channels:
mni_info = mne.create_info(ch_names=list(mni_montage.ch_names),
                           ch_types=[channels_types_mapping[ch] for ch in list(mni_montage.ch_names)],
                           sfreq=100)
# Add the montage:
mni_info.set_montage(mni_montage)
orientations = {
    "left": {"azimuth": 180, "elevation": None, "distance": 450, "focalpoint": "auto"},
    "front": {"azimuth": 90, "elevation": None, "distance": 450, "focalpoint": "auto"},
    "right": {"azimuth": 0, "elevation": None, "distance": 450, "focalpoint": "auto"},
    "back": {"azimuth": -90, "elevation": None, "distance": 450, "focalpoint": "auto"},
    "top": {"azimuth": 0, "elevation": 180, "distance": 450, "focalpoint": "auto"},
    "bottom": {"azimuth": 180, "elevation": 0, "distance": 450, "focalpoint": "auto"}
}

# Plot the ecog and seeg data separately:
for typ in ["ecog", "seeg"]:
    if typ == "ecog":
        # Render brain:
        brain = mne.viz.Brain(
            "fsaverage",
            subjects_dir=get_fs_directory(),
            surf="pial",
            cortex="low_contrast",
            alpha=1,
            background="white",
        )
        brain.add_sensors(mni_info, trans="fsaverage", ecog=True, seeg=False)
    else:
        brain = mne.viz.Brain(
            "fsaverage",
            subjects_dir=get_fs_directory(),
            surf="pial",
            cortex="low_contrast",
            alpha=0.4,
            background="white",
        )
        brain.add_sensors(mni_info, trans="fsaverage", ecog=False, seeg=True)

    # Plot the channels in different orientations:
    for i, ori in enumerate(orientations.values()):
        brain.show_view(**ori)
        brain.save_image(Path(save_path, "{}-{}.png".format(typ, i)))
    brain.close()

# ==================================================================================
# Coverage density plots:
labels_df = pd.concat(labels_df).reset_index(drop=True)
# For each channel, keep the first label that is not unknown:
labels_list = [
    next((part for part in s.split('/') if part not in ["Unknown"]), "Unknown")
    for s in labels_df["region"].to_list() if not pd.isna(s)
]
# Remove all the non-cortical labels:
labels_list = [lbl for lbl in labels_list if 'ctx_' in lbl]
# Get RGB values for each label:
labels_colors = count_colors(labels_list, "RdYlBu_r")


# Read the labels from this parcellation:
labels = mne.read_labels_from_annot("fsaverage", parc='aparc.a2009s', hemi='both', surf_name='white',
                                    annot_fname=None, regexp=None, subjects_dir=None, sort=True,
                                    verbose=None)

# ==========================================
# Inflated surf:
brain = mne.viz.Brain(
    "fsaverage",
    "both",
    "inflated",
    subjects_dir=get_fs_directory(),
    cortex="low_contrast",
    background="white",
    size=(800, 600),
)
# Loop through each count colors to plot the label in the corresponding color:
for lbl in labels_colors.keys():
    if "lh" in lbl:
        lbl_surf = lbl.replace("ctx_lh_", '') + "-lh"
    else:
        lbl_surf = lbl.replace("ctx_rh_", '') + "-rh"
    label = [lbl_ for lbl_ in labels if lbl_.name == lbl_surf][0]
    # Handle the string to match that of the parcellation:
    brain.add_label(label, borders=False, color=labels_colors[lbl])
# Plot the channels in different orientations:
for i, ori in enumerate(orientations.values()):
    brain.show_view(**ori)
    brain.save_image(Path(save_path, "{}_density_inflated.png".format(i)))

# Create a separate color bar:
cts = Counter(labels_list)
norm = Normalize(vmin=min(list(cts.values())), vmax=max(list(cts.values())))
cmap = plt.cm.RdYlBu_r
# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(1, 4))
fig.subplots_adjust(right=0.5)
# Create a ScalarMappable and initialize it with the normalization and colormap
sm = ScalarMappable(norm=norm, cmap=cmap)
# Create the colorbar
cbar = fig.colorbar(sm, cax=ax, orientation='vertical', aspect=50)
cbar.set_label('# of Elec.')
# Show the colorbar
plt.savefig(Path(save_path, "cbar.svg"))
