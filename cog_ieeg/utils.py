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
from pathlib import Path


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
