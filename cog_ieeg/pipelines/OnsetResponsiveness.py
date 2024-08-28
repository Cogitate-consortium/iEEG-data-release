import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from pingouin import ttest

from cog_ieeg.vizualization import plot_ieeg_image
from cog_ieeg.processing import baseline_scaling
from cog_ieeg.utils import get_pipeline_config

import environment_variables as ev


def run_onset_responsiveness(param, subjects,
                             plot_single_channels=True, plot_only_responsive=True):
    print("=" * 80)
    print("Welcome to Onset Responsiveness!")
    print("The onset responsive channels of the following subjects will be determined: ")
    print(subjects)
    if not isinstance(param, dict):
        print("Using the config file:")
        print(param)
    print("If you have the plotting options on, it will take some time!")
    if isinstance(subjects, str):
        subjects = [subjects]

    # Preallocate a variable to store the results of all participants:
    all_results = []
    # ======================================================================================================
    # Looping through each subject:
    for subject in subjects:
        print("=" * 60)
        print("Onset responsiveness {}".format(subject))

        # ======================================================================================================
        # Load the config and prepare directories:
        if not isinstance(param, dict):
            with open(param) as f:
                param = json.load(f)

        # Create path to save the data:
        save_root_results = Path(ev.bids_root, 'derivatives', 'onset_responsiveness',
                                 'sub-' + subject, 'ses-' + param["session"], param["data_type"],
                                 param["signal"], "results")
        save_root_figures = Path(ev.bids_root, 'derivatives', 'onset_responsiveness',
                                 'sub-' + subject, 'ses-' + param["session"], param["data_type"],
                                 param["signal"], "figures")
        if not os.path.isdir(save_root_results):
            # Creating the directory:
            os.makedirs(save_root_results)
        if not os.path.isdir(save_root_figures):
            # Creating the directory:
            os.makedirs(save_root_figures)
        # Create the file prefix:
        file_prefix = 'sub-{}_ses-{}_task-{}_desc-{}-{}'.format(subject, param["session"],
                                                                param["task"], "on_resp", param["data_type"])

        # Pre-allocate to store this subject results:
        subject_results = []

        # ======================================================================================================
        # Load and prepare the data:
        # Set path to the data:
        epochs_file = Path(ev.bids_root, 'derivatives', 'preprocessing',
                           'sub-' + subject, 'ses-' + param["session"], param["data_type"],
                           "epoching", param["signal"],
                           "sub-{}_ses-{}_task-{}_desc-epoching_{}-epo.fif".format(subject,
                                                                                   param["session"], param["task"],
                                                                                   param["data_type"]))
        # Load the epochs:
        epochs = mne.read_epochs(epochs_file, preload=True)
        # Pick the channels of interest:
        picks = mne.pick_types(epochs.info, **param["channel_types"])
        epochs.pick(picks)

        # Pick the conditions of interest:
        if param["condition"] is not None:
            epochs = epochs[param["condition"]]

        # Performing baseline correction if required:
        if param["baseline_method"] is not None:
            baseline_scaling(epochs, correction_method=param["baseline_method"],
                             baseline=param["baseline_time"])

        # ======================================================================================================
        # Prepare the data for the test:
        # This test consists of comparing the activation pre and post stimulus presentation
        # Therefore splitting the data in baseline and test window:
        prestim_epochs = epochs.copy().crop(tmin=param["prestimulus_window"][0],
                                            tmax=param["prestimulus_window"][1])
        poststim_epochs = epochs.copy().crop(tmin=param["poststimulus_window"][0],
                                             tmax=param["poststimulus_window"][1])

        # Loop through each channel:
        for ch in prestim_epochs.ch_names:
            # Extract this channel's data:
            ch_prestim = np.squeeze(prestim_epochs.get_data(picks=ch))
            ch_poststim = np.squeeze(poststim_epochs.get_data(picks=ch))
            # Different metric can be used, which might be better suited depending on the signal:
            if param["metric"] == "mean":
                ch_prestim = np.mean(ch_prestim, axis=1)
                ch_poststim = np.mean(ch_poststim, axis=1)
            elif param["metric"] == "auc":
                ch_prestim = np.trapz(ch_prestim, axis=1)
                ch_poststim = np.trapz(ch_poststim, axis=1)
            elif param["metric"] == "ptp":
                ch_prestim = np.ptp(ch_prestim, axis=1)
                ch_poststim = np.ptp(ch_poststim, axis=1)
            else:
                raise Exception("The metric passed is not supported, must be either mean, auc or ptp, check spelling!")

            # Test for statistical difference:
            ttest(ch_prestim, ch_poststim, paired=True, r=0.707, alternative='two-sided').round(3)
            ch_res = ttest(ch_prestim, ch_poststim, paired=True, r=0.707, alternative='two-sided').round(3)

            # Store the results in a dataframe:
            subject_results.append(pd.DataFrame({
                "subject": subject,
                "channel": "-".join([subject, ch]),
                "statistic": float(ch_res["T"].item()),
                "pvalue": float(ch_res["p-val"].item()),
                "bf": float(ch_res["BF10"].item()),
                "reject": float(ch_res["BF10"].item()) > param["bf_thresh"],
                "df": float(ch_res["dof"].item()),
                "lowCI": float(ch_res["CI95%"].item()[0]),
                "highCI": float(ch_res["CI95%"].item()[1]),
                "f_size": np.mean(ch_poststim - ch_prestim) / np.std(ch_poststim - ch_prestim)
            }, index=[0]))
        # Concatenate the subject's results:
        subject_results = pd.concat(subject_results).reset_index(drop=True)

        # Save the results:
        full_file_name = Path(save_root_results, '{}{}'.format(file_prefix, '-results.csv'))
        subject_results.to_csv(full_file_name, index=False)

        # Append to the rest:
        all_results.append(subject_results)

        # Plot the channels:
        if plot_single_channels:
            # Get the trials order:
            metadata = epochs.metadata
            metadata = metadata.reset_index(drop=True)
            metadata_reordered = metadata[param["column_order"]]
            trials_order = metadata_reordered.sort_values(by=param["column_order"]).index
            # First the significant ones:
            sig_channels = subject_results.loc[subject_results["reject"] == True, "channel"].to_list()
            sig_root = Path(save_root_figures, "sig")
            if not os.path.isdir(sig_root):
                # Creating the directory:
                os.makedirs(sig_root)
            for ch in sig_channels:
                plot_ieeg_image(epochs, ch.split("-")[1], show=False, 
                                units=param['units'], scalings=param['scaling'], cmap="RdYlBu_r",
                                center=None, ylim_prctile=None, logistic_cmap=True, ci=0.95)
                fig_file = Path(sig_root, '{}{}{}'.format(file_prefix, ch, '-image.png'))
                plt.savefig(fig_file)
                plt.close()

            # Then unsignificant ones:
            if not plot_only_responsive:
                non_sig_channels = subject_results.loc[subject_results["reject"] == False, "channel"].to_list()
                non_sig_root = Path(save_root_figures, "non_sig")
                if not os.path.isdir(non_sig_root):
                    # Creating the directory:
                    os.makedirs(non_sig_root)
                for ch in non_sig_channels:
                    plot_ieeg_image(epochs, ch.split("-")[1], show=False, 
                                    units=param['units'], scalings=param['scaling'], cmap="RdYlBu_r",
                                    center=None, ylim_prctile=None, logistic_cmap=True, ci=0.95)
                    fig_file = Path(non_sig_root, '{}{}{}'.format(file_prefix, ch, '-image.png'))
                    plt.savefig(fig_file)
                    plt.close()

    # Save all subjects results in the same directory:
    save_root_results = Path(ev.bids_root, 'derivatives', 'onset_responsiveness',
                             'sub-' + "all", 'ses-' + param["session"], param["data_type"],
                             param["signal"], "results")
    if not os.path.isdir(save_root_results):
        # Creating the directory:
        os.makedirs(save_root_results)
    # Concatenate the subject's results:
    all_results = pd.concat(all_results).reset_index(drop=True)

    # Save the results:
    file_prefix = 'sub-{}_ses-{}_task-{}_desc-{}'.format("all", param["session"],
                                                         param["task"], "on_resp")
    full_file_name = Path(save_root_results, '{}{}'.format(file_prefix, '-results.csv'))
    all_results.to_csv(full_file_name, index=False)

    return all_results


if __name__ == "__main__":
    config_file = get_pipeline_config('onset_responsiveness')
    subjects_list = ["CF102"]  # ["SF124", "SF125", "SF126"]
    run_onset_responsiveness(config_file, subjects_list,
                             plot_single_channels=True, plot_only_responsive=True)
