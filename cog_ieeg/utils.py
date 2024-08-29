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
    with pkg_resources.path('cog_ieeg.config', config_name) as config_path:
        return config_path


def load_config(config_file):
    with open(config_file, 'r') as configfile:
        return json.load(configfile)


def save_config(config, config_file):
    with open(config_file, 'w') as configfile:
        json.dump(config, configfile, indent=4)
    return None


def print_config(config):
    # Print the content of the configuration in a nicely formatted way
    print("Configuration content:\n")
    print(json.dumps(config, indent=4))
    return None


def print_config_path(config_file):
    # Ensure the config_file is a Path object for consistent formatting
    config_file = Path(config_file)
    # Print the full file path
    print(f"Configuration file path: {config_file.resolve()}\n")
    return None


def get_data_directory():
    system = platform.system()
    
    if system == "Windows":
        data_dir = Path(os.getenv('USERPROFILE')) / "Documents" / "COGITATE"
    elif system == "Darwin":  # macOS
        data_dir = Path.home() / "Library" / "Documents" / "COGITATE"
    else:  # Assume Linux or other UNIX-like systems
        data_dir = Path.home() / ".local" / "share"
    
    return data_dir


def create_default_config():
    # Define the default values
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
    # Save the configuration to a JSON file
    with open(get_config_path('config-path.json'), 'w') as configfile:
        json.dump(config, configfile, indent=4)

    return None


def set_bids_root(new_path, update_fs_dir=True):
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    config['Paths']['bids_root'] = str(new_path)
    if update_fs_dir:
        config['Paths']['fs_directory'] = str(Path(new_path) / 'derivatives' / 'fs')
    # Save the configuration to a JSON file
    save_config(config, get_config_path('config-path.json'))
    return None


def set_fs_directory(fs_directory):
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    config['XNAT']['fs_directory'] = fs_directory
    save_config(config, get_config_path('config-path.json'))
    return None


def set_xnat_host(xnat_host):
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    config['XNAT']['xnat_host'] = xnat_host
    save_config(config, get_config_path('config-path.json'))
    return None


def set_xnat_project(xnat_project):
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    config['XNAT']['xnat_project'] = xnat_project
    save_config(config, get_config_path('config-path.json'))
    return None


def get_bids_root():
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    return config['Paths']['bids_root']


def get_fs_directory():
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    return config['Paths']['fs_directory']


def get_xnat_host():
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    return config['XNAT']['xnat_host']


def get_xnat_project():
    with open(get_config_path('config-path.json'), 'r') as configfile:
        config = json.load(configfile)
    return config['XNAT']['xnat_project']


def get_pipeline_config(pipeline, preload=True):
    """
    Load a default configuration file from the 'config' directory within the package.

    :param pipeline: The name of the pipeline file (e.g., 'preprocessing_config-default.json').
    :param preload: The name of the pipeline file (e.g., 'preprocessing_config-default.json').
    :return: Dictionary containing the configuration.
    """
    assert pipeline.lower() in _KNOWN_PIPELINES, (f'pipeline must be {_KNOWN_PIPELINES}, \n'
                                                  f'Got {pipeline} instead')
    config = get_config_path(f'{pipeline.lower()}_config-default.json')
    if preload:
        return load_config(config)
    else:
        return config


def path_generator(directory):
    """
    Generate a folder if it doesn't exist.

    :param directory: string or pathlib path object, folder to create
    :return: directory path
    """
    #
    if not os.path.isdir(directory):
        # Creating the directory:
        os.makedirs(directory)

    return directory


def save_param(param, save_path, step, signal, file_prefix, file_extension):
    """
    Save the configuration parameters used to generate the data.

    :param param: dictionary, parameters used to generate the data
    :param save_path: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    """
    # Saving the config of this particular step:
    config_file_name = Path(save_path, '{}_desc-{}_ieeg{}.json'.format(file_prefix, step,
                                                                       file_extension.split('.')[0]))
    with open(str(config_file_name), 'w') as outfile:
        json.dump(param[step][signal], outfile)
    # Saving the entire config file as well:
    config_file_name = Path(save_path, '{}_desc-{}_ieeg{}.json'.format(file_prefix, 'all',
                                                                       file_extension.split('.')[0]))
    with open(str(config_file_name), 'w') as outfile:
        json.dump(param, outfile)

    return None


def mne_data_saver(data, param, save_root, step, signal, file_prefix, file_extension="-raw.fif"):
    """
    Save different instances of MNE objects.

    :param data: mne object (epochs, evoked, raw...), data to be saved
    :param param: dictionary, parameters used to generate the data
    :param save_root: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    :return: None
    """
    print("-" * 40)
    print("Saving mne object")

    # First, generating the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)
    # Generating the full file name:
    full_file_name = Path(save_path, '{}_desc-{}_ieeg{}'.format(file_prefix, step, file_extension))
    # Saving the data:
    data.save(full_file_name, overwrite=True)
    # Saving the config:
    save_param(param, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return None
