"""
This module provides utility functions to support various operations in intracranial EEG (iEEG) data analysis.

The utility functions in this module include file handling, path generation, and saving/loading configurations.

Author:
-------
Alex Lepauvre
Katarina Bendtz
Simon Henin

License:
--------
This code is licensed under the MIT License.
"""

import json
import os
import platform
from pathlib import Path
import importlib.resources as pkg_resources

_KNOWN_PIPELINES = ('preprocessing', 'preprocessing-doc', 'onset_responsiveness', 'decoding')


def get_config_path(config_name):
    """
    Get the path to the specified configuration file.

    Parameters
    ----------
    config_name : str
        The name of the configuration file.

    Returns
    -------
    config_path : pathlib.Path
        The path to the configuration file.
    """
    with pkg_resources.path('cog_ieeg.config', config_name) as config_path:
        return config_path


def load_config(config_file):
    """
    Load a configuration from a JSON file.

    Parameters
    ----------
    config_file : str or pathlib.Path
        The path to the configuration file.

    Returns
    -------
    dict
        The configuration as a dictionary.
    """
    with open(config_file, 'r') as configfile:
        return json.load(configfile)


def save_config(config, config_file):
    """
    Save a configuration to a JSON file.

    Parameters
    ----------
    config : dict
        The configuration to save.
    config_file : str or pathlib.Path
        The path to the file where the configuration should be saved.

    Returns
    -------
    None
    """
    with open(config_file, 'w') as configfile:
        json.dump(config, configfile, indent=4)
    return None


def print_config(config):
    """
    Print the configuration content in a nicely formatted way.

    Parameters
    ----------
    config : dict
        The configuration dictionary to print.

    Returns
    -------
    None
    """
    print("Configuration content:\n")
    print(json.dumps(config, indent=4))
    return None


def print_config_path(config_file):
    """
    Print the full path of the configuration file.

    Parameters
    ----------
    config_file : str or pathlib.Path
        The path to the configuration file.

    Returns
    -------
    None
    """
    config_file = Path(config_file)
    print(f"Configuration file path: {config_file.resolve()}\n")
    return None


def get_data_directory():
    """
    Get the default data directory based on the operating system.

    Returns
    -------
    pathlib.Path
        The path to the data directory.
    """
    system = platform.system()

    if system == "Windows":
        data_dir = Path(os.getenv('USERPROFILE')) / "Documents" / "COGITATE"
    elif system == "Darwin":  # macOS
        data_dir = Path.home() / "Library" / "Documents" / "COGITATE"
    else:  # Assume Linux or other UNIX-like systems
        data_dir = Path.home() / ".local" / "share"
    
    return data_dir


def create_default_config():
    """
    Create and save a default configuration file.

    The configuration includes default paths and XNAT settings.

    Returns
    -------
    None
    """
    data_dir = get_data_directory()
    bids_root = data_dir / 'bids'

    config = {
        "Paths": {
            "bids_root": str(bids_root),
            "fs_directory": str(bids_root / 'derivatives' / 'fs')
        },
        "XNAT": {
            "xnat_host": 'cogitate-data',
            "xnat_project": 'COG_IEEG_EXP1_BIDS'
        }
    }

    with open(get_config_path('config-path.json'), 'w') as configfile:
        json.dump(config, configfile, indent=4)

    if not os.path.isdir(config["Paths"]["bids_root"]):
        os.makedirs(config["Paths"]["bids_root"])
        os.makedirs(config["Paths"]["fs_directory"])

    return None


def set_bids_root(new_path, update_fs_dir=True):
    """
    Set a new BIDS root directory in the configuration.

    Parameters
    ----------
    new_path : str or pathlib.Path
        The new BIDS root directory.
    update_fs_dir : bool, optional
        Whether to update the FreeSurfer directory as well (default is True).

    Returns
    -------
    None
    """
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    config['Paths']['bids_root'] = str(new_path)
    if update_fs_dir:
        config['Paths']['fs_directory'] = str(Path(new_path) / 'derivatives' / 'fs')
    save_config(config, get_config_path('config-path.json'))
    return None


def set_fs_directory(fs_directory):
    """
    Set a new FreeSurfer directory in the configuration.

    Parameters
    ----------
    fs_directory : str or pathlib.Path
        The new FreeSurfer directory.

    Returns
    -------
    None
    """
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    config['XNAT']['fs_directory'] = fs_directory
    save_config(config, get_config_path('config-path.json'))
    return None


def set_xnat_host(xnat_host):
    """
    Set a new XNAT host in the configuration.

    Parameters
    ----------
    xnat_host : str
        The new XNAT host.

    Returns
    -------
    None
    """
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    config['XNAT']['xnat_host'] = xnat_host
    save_config(config, get_config_path('config-path.json'))
    return None


def set_xnat_project(xnat_project):
    """
    Set a new XNAT project in the configuration.

    Parameters
    ----------
    xnat_project : str
        The new XNAT project.

    Returns
    -------
    None
    """
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    config['XNAT']['xnat_project'] = xnat_project
    save_config(config, get_config_path('config-path.json'))
    return None


def get_bids_root():
    """
    Get the BIDS root directory from the configuration.

    Returns
    -------
    str
        The BIDS root directory.
    """
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    return config['Paths']['bids_root']


def get_fs_directory():
    """
    Get the FreeSurfer directory from the configuration.

    Returns
    -------
    str
        The FreeSurfer directory.
    """
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    return config['Paths']['fs_directory']


def get_xnat_host():
    """
    Get the XNAT host from the configuration.

    Returns
    -------
    str
        The XNAT host.
    """
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    return config['XNAT']['xnat_host']


def get_xnat_project():
    """
    Get the XNAT project from the configuration.

    Returns
    -------
    str
        The XNAT project.
    """
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    return config['XNAT']['xnat_project']


def get_pipeline_config(pipeline, preload=True):
    """
    Load a default configuration file for a specific pipeline.

    Parameters
    ----------
    pipeline : str
        The name of the pipeline (e.g., 'preprocessing').
    preload : bool, optional
        Whether to preload the configuration file (default is True).

    Returns
    -------
    dict or pathlib.Path
        The loaded configuration dictionary or the path to the configuration file.
    """
    assert pipeline.lower() in _KNOWN_PIPELINES, (
        f'pipeline must be {_KNOWN_PIPELINES}, \n'
        f'Got {pipeline} instead'
    )
    config = get_config_path(f'{pipeline.lower()}_config-default.json')
    if preload:
        return load_config(config)
    else:
        return config


def path_generator(directory):
    """
    Generate a folder if it doesn't exist.

    Parameters
    ----------
    directory : str or pathlib.Path
        The folder to create.

    Returns
    -------
    directory : pathlib.Path
        The path to the directory.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)

    return directory


def save_param(param, save_path, step, signal, file_prefix, file_extension):
    """
    Save the configuration parameters used to generate the data.

    Parameters
    ----------
    param : dict
        The parameters used to generate the data.
    save_path : str or pathlib.Path
        The path to where the data should be saved.
    step : str
        The name of the preprocessing step.
    signal : str
        The name of the signal being saved.
    file_prefix : str
        The prefix of the file to save.
    file_extension : str
        The file name extension.

    Returns
    -------
    None
    """
    config_file_name = Path(save_path, '{}_desc-{}_ieeg{}.json'.format(file_prefix, step,
                                                                       file_extension.split('.')[0]))
    with open(str(config_file_name), 'w') as outfile:
        json.dump(param[step][signal], outfile, indent=4)
    config_file_name = Path(save_path, '{}_desc-{}_ieeg{}.json'.format(file_prefix, 'all',
                                                                       file_extension.split('.')[0]))
    with open(str(config_file_name), 'w') as outfile:
        json.dump(param, outfile, indent=4)

    return None


def mne_data_saver(data, param, save_root, step, signal, file_prefix, file_extension="-raw.fif"):
    """
    Save different instances of MNE objects.

    Parameters
    ----------
    data : mne object
        Data to be saved (e.g., epochs, evoked, raw).
    param : dict
        The parameters used to generate the data.
    save_root : str or pathlib.Path
        The path to where the data should be saved.
    step : str
        The name of the preprocessing step.
    signal : str
        The name of the signal being saved.
    file_prefix : str
        The prefix of the file to save.
    file_extension : str, optional
        The file name extension (default is "-raw.fif").

    Returns
    -------
    None
    """
    print("-" * 40)
    print("Saving MNE object")

    save_path = Path(save_root, step, signal)
    path_generator(save_path)
    full_file_name = Path(save_path, '{}_desc-{}_ieeg{}'.format(file_prefix, step, file_extension))
    data.save(full_file_name, overwrite=True)
    save_param(param, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return None
