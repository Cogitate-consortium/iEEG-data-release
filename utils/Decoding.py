import json
import os
from pathlib import Path

import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from mne.decoding import (GeneralizingEstimator, cross_val_multiscore)

from utils.HelperFunctions import baseline_scaling
import environment_variables as ev


def decoding(param, subjects):
    print("=" * 80)
    print("Welcome to Decoding!")
    print("The onset responsive channels of the following subjects will be determined: ")
    print(subjects)
    if not isinstance(param, dict):
        print("Using the config file:")
        print(param)
    if isinstance(subjects, str):
        subjects = [subjects]

    # Preallocate a variable to store the results of all participants:
    subjects_scores = []
    # ======================================================================================================
    # Looping through each subject:
    for subject in subjects:
        print("=" * 60)
        print("Decoding {}".format(subject))

        # ======================================================================================================
        # Load the config and prepare directories:
        if not isinstance(param, dict):
            with open(param) as f:
                param = json.load(f)

        # Create path to save the data:
        save_root_results = Path(ev.bids_root, 'derivatives', 'decoding',
                                 'sub-' + subject, 'ses-' + param["session"], param["data_type"],
                                 param["signal"], "results")
        save_root_figures = Path(ev.bids_root, 'derivatives', 'decoding',
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
                                                                param["task"], "decoding", param["data_type"])

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
        if param["resample_freq"] is not None:
            epochs.resample(param["resample_freq"], n_jobs=param["njobs"])

        # ======================================================================================================
        # Prepare the classifier:
        # initialize classifier pipeline
        clf_steps = []
        if param['scaler']:
            clf_steps.append(StandardScaler())
        if param["do_feature_selection"]:
            clf_steps.append(SelectKBest(f_classif, k=param["k_features"]))
        clf_steps.append(svm.SVC(kernel='linear', class_weight='balanced'))
        clf = make_pipeline(*clf_steps)

        # Prepare the classifier:
        time_gen = GeneralizingEstimator(clf, n_jobs=param["njobs"], scoring=param["metric"],
                                         verbose="ERROR")

        # ======================================================================================================
        # Run the classifier:
        if param["train_group"] == param["test_group"]:
            # Checking that the cross fold validation is set to true:
            if param["n_folds"] is None:
                raise Exception("If you are doing within task decoding, you must use cross fold validation to be able"
                                "\nto test your trained decoding on something!")
            print("Decoding {} with {} folds stratified cross validation:".format(param["decoding_target"],
                                                                                  param["n_folds"]))
            # Extract the data:
            data = epochs.get_data(copy=True)
            # Get the classes:
            y = epochs.metadata[param["decoding_target"]].values
            # Run the decoding:
            scores = cross_val_multiscore(time_gen, data, y, cv=param["n_folds"], n_jobs=None)
        else:
            print("Decoding {}. Training on {} and generalizing to {}".format(param["decoding_target"],
                                                                              param["train_group"],
                                                                              param["test_group"]))
            # Extract the training data:
            train_epochs = epochs[param["train_condition"]].get_data()
            x_train = train_epochs.get_data(copy=True)
            y_train = train_epochs.metadata[param["decoding_target"]].values
            # Train the classifier:
            time_gen.fit(X=x_train, y=y_train)

            # Extract the training data:
            test_epochs = epochs[param["test_condition"]].get_data()
            x_test = test_epochs.get_data()
            y_test = test_epochs.metadata[param["decoding_target"]].values
            # Test the classifier (i.e. generalization):
            scores = time_gen.score(X=x_test, y=y_test)

        # ======================================================================================================
        # Plot the results:
        fig, ax = plt.subplots()
        # Plot matrix with transparency:
        im = ax.imshow(np.mean(scores, axis=0), cmap="RdYlBu_r",
                       extent=[epochs.times[0], epochs.times[-1], epochs.times[0], epochs.times[-1]],
                       origin="lower",  # aspect="equal",
                       interpolation="lanczos", vmin=0, vmax=1)
        # Add the axis labels and so on:
        ax.set_xlim([epochs.times[0], epochs.times[-1]])
        ax.set_ylim([epochs.times[0], epochs.times[-1]])
        ax.set_xlabel("Testing time (s)")
        ax.set_ylabel("Training time (s)")
        if param["train_group"] == param["test_group"]:
            ax.set_title("Decoding of {}".format(param["decoding_target"]))
        else:
            ax.set_title("Decoding of {}, \nTrain condition: {}, Test "
                         "condition: {}".format(param["decoding_target"],
                                                param["train_condition"],
                                                param["test_condition"]))
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig_file = Path(save_root_figures, '{}{}'.format(file_prefix, '-image.png'))
        plt.savefig(fig_file)
        plt.close()

        # Plot the time resolved decoding (i.e. the diagonal)
        fig, ax = plt.subplots()
        ax.plot(epochs.times, np.diag(np.mean(scores, axis=0)))
        ax.set_xlim([epochs.times[0], epochs.times[-1]])
        ax.set_ylim([0, 1])
        ax.axvline(0, color='k')
        ax.axhline(0.5, color=[0.5, 0.5, 0.5])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(param["metric"])
        fig_file = Path(save_root_figures, '{}{}'.format(file_prefix, '-time_resolved.png'))
        plt.savefig(fig_file)
        plt.close()

        # Save the scores:
        full_file_name = Path(save_root_results, '{}{}'.format(file_prefix, '-scores.npy'))
        np.save(full_file_name, scores)

        # Append to the rest:
        subjects_scores.append(np.mean(scores, axis=0))

    # Save all subjects results in a separate directory:
    save_root_results = Path(ev.bids_root, 'derivatives', 'decoding',
                             'sub-' + "all", 'ses-' + param["session"], param["data_type"],
                             param["signal"], "results")
    if not os.path.isdir(save_root_results):
        # Creating the directory:
        os.makedirs(save_root_results)
    file_prefix = 'sub-{}_ses-{}_task-{}_desc-{}'.format("all", param["session"],
                                                         param["task"], "decoding")
    full_file_name = Path(save_root_results, '{}{}'.format(file_prefix, '-scores.npy'))
    np.save(full_file_name, subjects_scores)

    return subjects_scores


if __name__ == "__main__":
    config_file = r"../configs/Decoding_config-default.json"
    subjects_list = ["SF102"]
    decoding(config_file, subjects_list)
