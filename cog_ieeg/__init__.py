from .localization import create_montage, get_roi_channels

from .vizualization import get_cmap_rgb_values, plot_ieeg_image

from .xnat_utilities import xnat_download

from .utils import create_default_config, get_config_path

import importlib.resources as pkg_resources

from pathlib import Path


def initialize_config():
    config_path = get_config_path('config-path.json')
    
    # Check if the config file already exists
    if not Path(config_path).exists():
        create_default_config()

initialize_config()
