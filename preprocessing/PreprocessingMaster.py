""" This script does pre-processing of ECoG data
    authors: Alex Lepauvre and Katarina Bendtz
    alex.lepauvre@ae.mpg.de
    katarina.bendtz@tch.harvard.edu
    Dec 2020
"""
import argparse
import time
from mne_bids import BIDSPath, read_raw_bids

from preprocessing.PreprocessingParametersClass import PreprocessingParameters
from preprocessing.SubjectInfo import SubjectInfo
from HelperFunctions import *
from general_helper_functions.data_general_utilities import (set_annotations,
                                                             set_channels_types,
                                                             set_montage,
                                                             baseline_scaling)
from general_helper_functions.plotters import (plot_ordered_epochs,
                                               plot_epochs_grids)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SUPPORTED_STEPS = [
    "notch_filtering",
    "automated_bad_channels_rejection",
    "manual_artifact_detection",
    "car",
    "laplace_reference",
    "frequency_bands_computations",
    "manual_artifact_detection",
    "epoching",
    "automated_artifact_detection",
    "plot_epochs"
]

ERROR_UNKNOWN_STEP_TEXT = "You have given the preprocessing step: {step} in the analysis paramters json file that is " \
                          "not \nsupported. The supported steps are those: " \
                          "{supported_steps}. " \
                          "\nMake sure you check the spelling in the analysis parameter json file!"

ERROR_RAW_MISSING = "You have called {step} after calling epoching. This step only works if " \
                    "\n the signal is continuous. Make sure you set the order of your " \
                    "\npreprocessing steps such that this step is called BEFORE you call epoching"
ERROR_EPOCHS_MISSING = "You have called {step} before calling epoching. This step only works if " \
                       "\n the signal is epoched already. Make sure you set the order of your " \
                       "\npreprocessing steps such that this step is called AFTER you call epoching"
ERROR_SIGNAL_MISSING = "For the preprocessing step {step}, you have passed the signal {signal} " \
                       "\nwhich does not exist at this stage of the preprocessing. Either you have asked " \
                       "\nfor that signal to be generated later OR you haven't asked for it to be " \
                       "\ngenerated. Make sure to check your config file"


def preprocessing():
    start = time.time()

    # ------------------------------------------------------------------------------------------------------------------
    # Setting the parameters
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('config', type=str, default=None,
                        help="Config file for analysis parameters (path and filename)")
    parser.add_argument('--subject', type=str, default=None,
                        help="subject ID, for instance SE101 or SF101")
    parser.add_argument('--interactive', action="store_true",
                        help="Option to preprocessing the preprocessing script interactively, allowing for manual input")
    args = parser.parse_args()
    print("-" * 40)
    print("Welcome to Preprocessing.py!")

    # Load the preprocessing parameters
    preprocessing_parameters = PreprocessingParameters(
        args.config)

    # Initializing the subject info:
    subject_info = SubjectInfo(args.subject, preprocessing_parameters, interactive=args.interactive)

    # ------------------------------------------------------------------------------------------------------------------
    # Preparing the data:
    # ------------------------------------------------------------------------------------------------------------------
    # Creating the bids path object:
    bids_path = BIDSPath(root=subject_info.BIDS_ROOT, subject=subject_info.SUBJ_ID,
                         session=preprocessing_parameters.session,
                         datatype=preprocessing_parameters.data_type,
                         task=preprocessing_parameters.task_name)
    # Loading the data under the term broadband, as it is what they are as long as no further filtering was employed
    raw = {"broadband": read_raw_bids(bids_path=bids_path, verbose=True)}
    try:
        # Setting the channel type based on the channels tsv, because there is a bug in the read raw bids function:
        raw["broadband"] = set_channels_types(raw["broadband"], bids_path)
    except IndexError:
        print("Did not find any channels.tsv to set the channel type. The channels type found in the file will be kept")
    # Loading the annotations, because if we have both eeg and ieeg data in the data sets, the mne bids reader is buggy:
    raw["broadband"] = set_annotations(raw["broadband"], bids_path)
    # Set the montage according to the expected coordinate space from the config:
    if preprocessing_parameters.montage_space is not None:
        try:
            raw["broadband"] = set_montage(raw["broadband"], bids_path,
                                           montage_space=preprocessing_parameters.montage_space)
        except IndexError:
            print("The eletrode reconstruction is missing for this subject! You will need to rerung it once you"
                  "have it!")
    # Convert the channels to mni space:
    if preprocessing_parameters.montage_space == "T1":
        mni_coords = ieeg_t12mni(raw["broadband"], preprocessing_parameters.fs_dir, "sub-" + subject_info.SUBJ_ID,
                                 template='fsaverage_sym', ch_types=None)
        # Save the MNI coordinates to the bids directory:
        mni_coords_file = Path(subject_info.BIDS_ROOT, "sub-" + subject_info.SUBJ_ID,
                               "ses-" + preprocessing_parameters.session,
                               preprocessing_parameters.data_type,
                               "sub-{}_ses-{}_space-fsaverage_electrodes.tsv".format(subject_info.SUBJ_ID,
                                                                                     preprocessing_parameters.session))
        # Save the file as a tsv:
        mni_coords.to_csv(mni_coords_file, sep='\t', index_label="name")

    # In debug mode, only a few channels + trigger channel are loaded
    if preprocessing_parameters.debug is True:
        # Dropping most channels but the first 4
        raw["broadband"].drop_channels(
            [ch for ch in raw["broadband"].info['ch_names'][10:-1] if ch != subject_info.TRIGGER_CHANNEL
             and ch != subject_info.TRIGGER_REF_CHANNEL])
        # Loading only the few channels to cut down loading time
        raw["broadband"].load_data()
    else:
        # Otherwise, loading everything
        raw["broadband"].load_data()

    # Downsampling the signal, otherwise things take forever:
    print("Downsampling the signal to 512Hz, this may take a little while")
    raw["broadband"].resample(
        512, n_jobs=preprocessing_parameters.njobs, verbose=True)
    print(raw["broadband"].info)

    print("Detrending the data")
    raw["broadband"].apply_function(lambda ch: ch - np.mean(ch), n_jobs=preprocessing_parameters.njobs,
                                    channel_wise=True)

    # Create the events in the signal from the annotation for later use:
    print('Creating annotations')
    events_from_annot, event_dict = mne.events_from_annotations(
        raw["broadband"])
    # And then deleting the annotations, not needed anymore. Makes interactive plotting stuff more reactive:
    raw["broadband"].set_annotations(
        mne.Annotations(onset=[], duration=[], description=[]))
    # Try to load the interruption landmark from the config file:
    if subject_info.interruption_index is None and preprocessing_parameters.interuption_landmark is not None \
            and args.interactive:
        subject_info.interruption_index = \
            find_interruption_index(
                events_from_annot, event_dict, preprocessing_parameters.interuption_landmark)
    elif subject_info.interruption_index is None and preprocessing_parameters.interuption_landmark is None:
        subject_info.interruption_index = False
    # Update the subject info:
    if preprocessing_parameters.save_output:
        subject_info.update_json()

    # ==================================================================================================================
    # Preprocessing loop
    # ==================================================================================================================
    for step in preprocessing_parameters.preprocessing_steps:
        # --------------------------------------------------------------------------------------------------------------
        # Notch filter:
        # --------------------------------------------------------------------------------------------------------------
        if step.lower() == "notch_filtering":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            # Looping through the different signals that are requested to be filtered:
            for ind, signal in enumerate(list(step_parameters.keys())):
                if 'raw' in locals() and signal in raw:
                    # Plotting the channels PSD before the notch filtering to evaluate the change:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print(
                            "Plotting the channels power spectral density before notch filtering")
                        plot_channels_psd(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal="broadband",
                                          file_extension=".png", plot_single_channels=False,
                                          channels_type=step_parameters[signal]["channel_types"])
                    # Filtering the signal:
                    raw[signal] = \
                        notch_filtering(raw[signal],
                                        njobs=preprocessing_parameters.njobs,
                                        **step_parameters[signal])
                    # Adding the step to the preprocessing parameters to log that it has been completed:
                    if ind == 0:
                        preprocessing_parameters.add_performed_analysis(step)
                    # Plotting the psd again to show the changes:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print(
                            "Plotting the channels power spectral density after notch filtering")

                        plot_channels_psd(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal=signal,
                                          file_extension=".png", plot_single_channels=False,
                                          channels_type=step_parameters[signal]["channel_types"])
                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                        mne_data_saver(raw[signal], preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-raw.fif")
                elif 'raw' not in locals():
                    raise Exception(ERROR_RAW_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(
                        step=step, signal=signal))

        # --------------------------------------------------------------------------------------------------------------
        # automated bad channels rejection:
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "automated_bad_channels_rejection":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            for ind, signal in enumerate(list(step_parameters.keys())):
                if 'raw' in locals() and signal in raw:
                    print("-" * 40)
                    print(
                        "Plotting the channels power spectral density after automated bad channels rejection")
                    plot_channels_psd(raw[signal],
                                      preprocessing_parameters,
                                      subject_info, step.lower(),
                                      data_type=preprocessing_parameters.data_type,
                                      signal=signal,
                                      file_extension=".png", plot_single_channels=True)
                    # Applying automated bad channels rejection:
                    raw[signal], detected_bad_channels = automated_bad_channel_detection(raw[signal],
                                                                                         **step_parameters[signal])
                    # Adding detected bad channels to the json:
                    subject_info.auto_bad_channels = detected_bad_channels
                    # Save bad channels to the channels tsv:
                    annotate_channels_tsv(bids_path, detected_bad_channels, "automated_rejection", overwrite=True)
                    # Keeping track of performed steps:
                    if ind == 0:
                        preprocessing_parameters.add_performed_analysis(step)
                    # Plotting the psd again to show the changes:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print(
                            "Plotting the channels power spectral density after automated bad channels rejection")
                        plot_bad_channels(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal="broadband",
                                          file_extension=".png", plot_single_channels=False)

                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                        mne_data_saver(raw[signal], preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-raw.fif")
                elif 'raw' not in locals():
                    raise Exception(ERROR_RAW_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(
                        step=step, signal=signal))

        elif step.lower() == "description_bad_channels_rejection":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            for ind, signal in enumerate(list(step_parameters.keys())):
                if 'raw' in locals() and signal in raw:
                    raw[signal], bad_channels = \
                        description_ch_rejection(
                            raw[signal], bids_path,
                            step_parameters[signal]["bad_channels_description"], subject_info)
                    if ind == 0:
                        preprocessing_parameters.add_performed_analysis(step)
                    if preprocessing_parameters.save_output:
                        subject_info.update_json()
                    if len(bad_channels) > 0:
                        # Plotting the psd again to show the changes:
                        if preprocessing_parameters.save_output:
                            print("-" * 40)
                            print(
                                "Plotting the channels power spectral density after manual bad channels rejection")
                            plot_bad_channels(raw[signal],
                                              preprocessing_parameters,
                                              subject_info, step.lower(),
                                              data_type=preprocessing_parameters.data_type, picks=bad_channels,
                                              signal="broadband",
                                              file_extension=".png", plot_single_channels=False)
                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                        mne_data_saver(raw[signal], preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-raw.fif")
                elif 'raw' not in locals():
                    raise Exception(ERROR_RAW_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(
                        step=step, signal=signal))

        # --------------------------------------------------------------------------------------------------------------
        # manual bad channels rejection:
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "manual_bad_channels_rejection":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            # This function is not possible if the script is set to be preprocessing non-interactively!
            if not args.interactive:
                raise Exception("You have asked to perform manual_bad_channels_rejection in non-interactive mode."
                                "\nThis is not possible! You should either set the interactive variable to true in "
                                "\nwhen calling Preprocessing.py OR remove this step from the preprocessing"
                                "steps list")
            for ind, signal in enumerate(list(step_parameters.keys())):
                if 'raw' in locals() and signal in raw:
                    raw[signal] = \
                        manual_signal_inspection(raw[signal], subject_info,
                                                 instructions="Select bad channels manually by clicking on them!")
                    if ind == 0:
                        preprocessing_parameters.add_performed_analysis(step)
                    if preprocessing_parameters.save_output:
                        subject_info.update_json()
                    # Plotting the psd again to show the changes:
                    # Plotting the psd again to show the changes:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print(
                            "Plotting the channels power spectral density after manual bad channels rejection")
                        plot_bad_channels(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal="broadband",
                                          file_extension=".png", plot_single_channels=False)
                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                        mne_data_saver(raw[signal], preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-raw.fif")
                elif 'raw' not in locals():
                    raise Exception(ERROR_RAW_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(
                        step=step, signal=signal))

        # --------------------------------------------------------------------------------------------------------------
        # Common average referencing:
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "car":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            # Looping through the signals we want to perform the CAR on:
            for ind, signal in enumerate(list(step_parameters.keys())):
                if 'raw' in locals() and signal in raw:
                    # Plotting the channels PSD before the CAR to see changes:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print(
                            "Plotting the channels power spectral density before common average referencing")
                        plot_channels_psd(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal=signal,
                                          file_extension=".png", plot_single_channels=False,
                                          channels_type=step_parameters[signal]["target_channel_types"])
                    raw[signal] = custom_car(
                        raw[signal], **step_parameters[signal])

                    # Adding the step to the preprocessing parameters to log that it has been completed:
                    if ind == 0:
                        preprocessing_parameters.add_performed_analysis(step)
                    # Plotting the psd again to show the changes:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print(
                            "Plotting the channels power spectral density after common average referencing")
                        plot_channels_psd(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal=signal,
                                          file_extension=".png", plot_single_channels=False,
                                          channels_type=step_parameters[signal]["target_channel_types"])
                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                        mne_data_saver(raw[signal], preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-raw.fif")
                elif 'raw' not in locals():
                    raise Exception(ERROR_RAW_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(
                        step=step, signal=signal))

        # --------------------------------------------------------------------------------------------------------------
        # Laplacian referencing:
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "laplace_reference":
            print("-" * 60)
            print("Performing " + step)
            # Looping through the signals we want to perform the laplace reference on:
            step_parameters = getattr(preprocessing_parameters, step)
            for ind, signal in enumerate(list(step_parameters.keys())):
                if 'raw' in locals() and signal in raw:
                    # Ploting the channels PSD before the laplace referencing to see changes:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print("Plotting the channels power spectral density before laplace referencing")
                        plot_channels_psd(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal=signal,
                                          file_extension=".png", plot_single_channels=False,
                                          channels_type=step_parameters[signal]["channel_types"])
                    # Generate the laplace mapping file name:
                    laplace_mapping_file = "sub-{0}_ses-" + preprocessing_parameters.session + \
                                           "_laplace_mapping_" + preprocessing_parameters.data_type + ".json"
                    sub_laplace_mapping = Path(bids_path.directory,
                                               laplace_mapping_file.format(subject_info.SUBJ_ID))
                    if sub_laplace_mapping.is_file():
                        with open(sub_laplace_mapping) as f:
                            laplacian_mapping = json.load(f)
                    else:  # If this is missing, the laplace mapping cannot be performed!
                        raise FileNotFoundError("The laplace mapping json is missing: {0}"
                                                "\nYou need to generate a laplace mapping json file before you can preprocessing "
                                                "this"
                                                "\nspecific step."
                                                "\nCheck out the script semi_automated_laplace_mapping.py".
                                                format(sub_laplace_mapping))
                    # Performing the laplace referencing:
                    raw[signal], reference_mapping, bad_channels = \
                        laplacian_referencing(raw[signal], laplacian_mapping,
                                              subjects_dir=preprocessing_parameters.fs_dir,
                                              subject="sub-" + args.subject,
                                              montage_space=preprocessing_parameters.montage_space,
                                              **step_parameters[signal])
                    # Adding the step to the preprocessing parameters to log that it has been completed:
                    if ind == 0:
                        preprocessing_parameters.add_performed_analysis(step)
                    # Handle the bad channels induced by the laplace referencing:
                    subject_info.laplace_bad_channels = [channel for channel in bad_channels
                                                         if channel not in raw[signal].info["bads"]]
                    subject_info.laplace_bad_channels_prop = len(subject_info.laplace_bad_channels) / \
                                                             len(raw[signal].info["ch_names"])
                    if preprocessing_parameters.save_output:
                        subject_info.update_json()
                    # Plotting the psd again to show the changes:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print("Plotting the channels power spectral density after laplace referencing")
                        plot_channels_psd(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal=signal,
                                          file_extension=".png", plot_single_channels=False,
                                          channels_type=step_parameters[signal]["channel_types"])
                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                        mne_data_saver(raw[signal], preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-raw.fif")

                    # Convert the relocalized channels to mni and save to tsv:
                    if preprocessing_parameters.montage_space == "T1":
                        mni_coords = ieeg_t12mni(raw[signal], preprocessing_parameters.fs_dir,
                                                 "sub-" + subject_info.SUBJ_ID,
                                                 template='fsaverage_sym', ch_types=None)
                        # Save the MNI coordinates to the bids directory:
                        mne_coords_file_root = \
                            path_generator(Path(subject_info.BIDS_ROOT, "derivatives", "preprocessing",
                                                "sub-" + subject_info.SUBJ_ID,
                                                "ses-" + preprocessing_parameters.session,
                                                preprocessing_parameters.data_type),
                                           step.lower(), signal=signal,
                                           previous_steps_list=preprocessing_parameters.analyses_performed,
                                           figure=False)
                        mni_coords_file = Path(mne_coords_file_root,
                                               "sub-{}_ses-{}_space-fsaverage_electrodes.tsv".format(
                                                   subject_info.SUBJ_ID,
                                                   preprocessing_parameters.session))
                        # Save the file as a tsv:
                        mni_coords.to_csv(mni_coords_file, sep='\t', index_label="name")

                elif 'raw' not in locals():
                    raise Exception(ERROR_RAW_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(step=step, signal=signal))

        # --------------------------------------------------------------------------------------------------------------
        # manual_artifact_detection
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "manual_artifact_detection":
            print("-" * 60)
            print("Performing " + step)
            # Looping through the signals we want to perform the laplace reference on:
            step_parameters = getattr(preprocessing_parameters, step)
            for ind, signal in enumerate(list(step_parameters.keys())):
                if 'raw' in locals() and signal in raw:
                    raw[signal] = \
                        manual_signal_inspection(raw[signal], subject_info,
                                                 instructions="Annotate bad segments of the data")
                    if ind == 0:
                        preprocessing_parameters.add_performed_analysis(step)
                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                        mne_data_saver(raw[signal], preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-raw.fif")
                elif 'raw' not in locals():
                    raise Exception(ERROR_RAW_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(
                        step=step, signal=signal))

        # --------------------------------------------------------------------------------------------------------------
        # high_gamma_computations
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "frequency_bands_computations":
            print("-" * 60)
            print("Performing " + step)
            # Looping through the signals we want to perform the laplace reference on:
            step_parameters = getattr(preprocessing_parameters, step)
            if 'raw' in locals():
                # Looping through the different signals bands to compute:
                for signal in step_parameters.keys():
                    # Extract the parameters for this specific frequency band:
                    frequency_band_parameters = step_parameters[signal]
                    # Compute the signal accordingly:
                    raw[signal] = \
                        frequency_bands_computations(
                            raw[frequency_band_parameters["source_signal"]
                            ], interruption_index=subject_info.interruption_index,
                            njobs=preprocessing_parameters.njobs,
                            **frequency_band_parameters["computation_parameters"])

                    # Plotting the psd:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print("Plotting the channels power spectral density of the specific signal")
                        plot_channels_psd(raw[signal],
                                          preprocessing_parameters,
                                          subject_info, step.lower(),
                                          data_type=preprocessing_parameters.data_type,
                                          signal=signal,
                                          file_extension=".png", plot_single_channels=False,
                                          channels_type=step_parameters[
                                              signal]["computation_parameters"]["channel_types"])

                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                        mne_data_saver(
                            raw[signal],
                            preprocessing_parameters,
                            subject_info,
                            step.lower(),
                            data_type=preprocessing_parameters.data_type,
                            signal=signal,
                            mne_file_extension="-raw.fif")
            elif 'raw' not in locals():
                raise Exception(ERROR_RAW_MISSING.format(step=step))

        # --------------------------------------------------------------------------------------------------------------
        # erp_computations
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "erp_computations":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            if 'raw' in locals():
                # Compute the high gamma:
                raw[step_parameters["signal_name"]] = \
                    erp_computation(
                        raw[step_parameters["source_signal"]],
                        njobs=preprocessing_parameters.njobs,
                        **step_parameters["computation_parameters"])

                # Plotting the psd:
                if preprocessing_parameters.save_output:
                    print("-" * 40)
                    print("Plotting the channels power spectral density of the ERP")
                    plot_channels_psd(raw[step_parameters["signal_name"]],
                                      preprocessing_parameters,
                                      subject_info, step.lower(),
                                      data_type=preprocessing_parameters.data_type,
                                      signal="broadband",
                                      file_extension=".png", plot_single_channels=False,
                                      channels_type=step_parameters["computation_parameters"]["channel_types"])

                # Saving the data:
                if preprocessing_parameters.save_intermediary_steps and preprocessing_parameters.save_output:
                    mne_data_saver(
                        raw[step_parameters["signal_name"]],
                        preprocessing_parameters,
                        subject_info,
                        step.lower(),
                        data_type=preprocessing_parameters.data_type,
                        signal=step_parameters["signal_name"],
                        mne_file_extension="-raw.fif")
            elif 'raw' not in locals():
                raise Exception(ERROR_RAW_MISSING.format(step=step))

        # --------------------------------------------------------------------------------------------------------------
        # epoching
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "epoching":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            # The epochs will be stored in a dictionary. Needs to be created first:
            epochs = {}
            for signal in list(step_parameters.keys()):
                print(signal)
                if 'raw' in locals() and signal in raw:
                    epochs[signal] = epoching(raw[signal], events_from_annot, event_dict,
                                              **step_parameters[signal])
                    # Saving the data:
                    if preprocessing_parameters.save_intermediary_steps or \
                            preprocessing_parameters.preprocessing_steps[-1] == step and \
                            preprocessing_parameters.save_output:
                        mne_data_saver(epochs[signal],
                                       preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-epo.fif")
                elif 'raw' not in locals():
                    raise Exception(ERROR_RAW_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(
                        step=step, signal=signal))

            # Deleting the raw after we did the epoching. This is to avoid the case where the user has set a step that
            # is to be performed on the non segmented data after the epoching was done.
            del raw

        # --------------------------------------------------------------------------------------------------------------
        # automated_artifact_detection
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "automated_artifact_detection":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            for ind, signal in enumerate(list(step_parameters.keys())):
                if 'epochs' in locals() and signal in epochs:
                    epochs[signal], ind_trials_to_drop = \
                        automated_artifact_detection(epochs[signal],
                                                     **step_parameters[signal][
                                                         "automated_artifact_rejection_parameters"])
                    # Only if we are on the first iteration of the signal loop do we need to update the analyses
                    # performed
                    if ind == 0:
                        preprocessing_parameters.add_performed_analysis(step)
                    # Plotting the trials that were rejected:
                    if preprocessing_parameters.save_output:
                        print("-" * 40)
                        print("Plotting the rejected trials info")
                        plot_rejected_trials(epochs[signal], ind_trials_to_drop,
                                             preprocessing_parameters,
                                             subject_info, step.lower(),
                                             data_type=preprocessing_parameters.data_type,
                                             signal=signal,
                                             file_extension=".png", plot_single_channels=True)
                    # Saving the data if it is the last step or if saving intermediary step is required:
                    if preprocessing_parameters.save_intermediary_steps or \
                            preprocessing_parameters.preprocessing_steps[-1] == step and \
                            preprocessing_parameters.save_output:
                        mne_data_saver(epochs[signal],
                                       preprocessing_parameters,
                                       subject_info,
                                       step.lower(),
                                       data_type=preprocessing_parameters.data_type,
                                       signal=signal,
                                       mne_file_extension="-epo.fif")
                elif 'raw' not in locals():
                    raise Exception(ERROR_EPOCHS_MISSING.format(step=step))
                elif signal not in raw:
                    raise Exception(ERROR_SIGNAL_MISSING.format(
                        step=step, signal=signal))

        # --------------------------------------------------------------------------------------------------------------
        # atlas_mapping
        # --------------------------------------------------------------------------------------------------------------
        elif step.lower() == "atlas_mapping":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            # First, relocating the free surfer directory:
            subject_free_surfer_dir = Path(
                preprocessing_parameters.fs_dir, "sub-" + args.subject)
            if not subject_free_surfer_dir.is_dir():
                if step_parameters["copy_recon"]:
                    source = Path(preprocessing_parameters.raw_root,
                                  step_parameters["recon_source"].format(args.subject))
                    try:
                        relocate_fs_folder(source, subject_free_surfer_dir)
                    except FileNotFoundError:
                        print("The free surfer reconstruction is absent for this subject!")
                        continue

            # Plotting the electrodes localization:
            if "raw" in locals():
                # Doing the probabilistic mapping onto the different atlases:
                # Getting the ROI again:
                electrodes_mapping_df = \
                    roi_mapping(raw["broadband"], step_parameters["list_parcellations"],
                                "sub-" + args.subject,
                                preprocessing_parameters.fs_dir, step,
                                subject_info, preprocessing_parameters)

                # Annotating the channels that sit outside the brain:
                # Get a list of the channels that are outside the brain:
                roi_df = electrodes_mapping_df[list(electrodes_mapping_df.keys())[0]]
                bad_channels = []
                # Looping through each channel to see if the only label is "unknown":
                for channel in list(roi_df["channel"]):
                    # Get the rois of this channel:
                    ch_roi = roi_df.loc[roi_df["channel"] == channel, "region"].item()
                    # Check whether this channel is labelled only as "unknow":
                    if len(ch_roi.split("/")) == 1:
                        if ch_roi.split("/")[0].lower() == "unknown":
                            bad_channels.append(channel)
                if len(bad_channels) > 0:
                    print("WARNING: The following channels were found to sit outside the brain!")
                    print(bad_channels)
                    print("These channels will be annotated in the bids channel tsv as sitting outside the brain!")
                    bad_channels_to_bids(args.subject, bids_path, bad_channels, description="outside_brain")
                if args.interactive:
                    # Plotting the electrodes localization on the brain surface
                    plot_electrode_localization(raw["broadband"].copy(), subject_info, preprocessing_parameters, step,
                                                subject_id="sub-" + args.subject,
                                                fs_subjects_directory=preprocessing_parameters.fs_dir,
                                                data_type=preprocessing_parameters.data_type,
                                                file_extension=".png", channels_to_plot=["ecog", "seeg"],
                                                montage_space=preprocessing_parameters.montage_space)
                    plot_electrode_localization(raw["broadband"].copy(), subject_info, preprocessing_parameters, step,
                                                subject_id="sub-" + args.subject,
                                                fs_subjects_directory=preprocessing_parameters.fs_dir,
                                                data_type=preprocessing_parameters.data_type,
                                                file_extension=".png", channels_to_plot=["ecog", "seeg"],
                                                montage_space=preprocessing_parameters.montage_space,
                                                plot_elec_name=True)
                else:
                    mne.utils.warn(
                        'You are running the preprocessing in non-interactive mode. The plotting of electrodes on '
                        'brain surface will be skipped because it requires interactive mode', RuntimeWarning)

            elif "epochs" in locals():
                # Doing the probabilistic mapping onto the different atlases:
                try:
                    electrodes_mapping_df = \
                        roi_mapping(epochs["broadband"], step_parameters["list_parcellations"],
                                    "sub-" + args.subject,
                                    preprocessing_parameters.fs_dir, step,
                                    subject_info, preprocessing_parameters)
                except RuntimeError:
                    print("WARNING: The montage was in head, but needs to be in MRI space to do the roi mapping!")
                    print("Converting back to MRI")
                    # Computing the trans from head back to fMRI
                    trans_head_mri = mne.coreg.estimate_head_mri_t("sub-" + args.subject,
                                                                   subjects_dir=preprocessing_parameters.fs_dir,
                                                                   verbose=None)
                    # Get the current montage
                    montage_mri = raw["broadband"].get_montage()
                    # Convert it from head to fmri
                    montage_mri.apply_trans(trans_head_mri)
                    # Remove the fiducials:
                    montage_mri.remove_fiducials()
                    # Apply it to the raw:
                    raw["broadband"] = raw["broadband"].set_montage(montage_mri)
                    # Getting the ROI again:
                    electrodes_mapping_df = \
                        roi_mapping(raw["broadband"], step_parameters["list_parcellations"],
                                    "sub-" + args.subject,
                                    preprocessing_parameters.fs_dir, step,
                                    subject_info, preprocessing_parameters)
                    # Annotating the channels that sit outside the brain:
                    # Get a list of the channels that are outside the brain:
                    roi_df = electrodes_mapping_df[list(electrodes_mapping_df.keys())[0]]
                    bad_channels = []
                    # Looping through each channel to see if the only label is "unknown":
                    for channel in list(roi_df["channel"]):
                        # Get the rois of this channel:
                        ch_roi = roi_df.loc[roi_df["channel"] == channel, "region"].item()
                        # Check whether this channel is labelled only as "unknow":
                        if len(ch_roi.split("/")) == 1:
                            if ch_roi.split("/")[0].lower() == "unknown":
                                bad_channels.append(channel)
                    if len(bad_channels) > 0:
                        print("WARNING: The following channels were found to sit outside the brain!")
                        print(bad_channels)
                        print("These channels will be annotated in the bids channel tsv as sitting outside the brain!")
                        bad_channels_to_bids(args.subject, bids_path, bad_channels, description="outside_brain")
                if args.interactive:
                    # Plotting the electrodes localization on the brain surface
                    plot_electrode_localization(epochs["broadband"].copy(), subject_info, preprocessing_parameters,
                                                step, subject_id="sub-" + args.subject,
                                                fs_subjects_directory=preprocessing_parameters.fs_dir,
                                                data_type=preprocessing_parameters.data_type,
                                                file_extension=".png", channels_to_plot=["ecog", "seeg"],
                                                montage_space=preprocessing_parameters.montage_space)
                else:
                    mne.utils.warn(
                        'You are running the preprocessing in non-interactive mode. The plotting of electrodes on '
                        'brain surface will be skipped because it requires interactive mode', RuntimeWarning)

        # Plotting the epoched data:
        elif step.lower() == "plot_epochs":
            print("-" * 60)
            print("Performing " + step)
            # Get the parameters of this specific step:
            step_parameters = getattr(preprocessing_parameters, step)
            # For ease of sharing across colleagues, the data are saved in a centralized folder for all participants.
            save_root = Path(bids_path.root, "derivatives",
                             "data_vizualization")
            if not os.path.isdir(save_root):
                os.mkdir(save_root)
            # Looping through each signal:
            for signal in step_parameters["signals"]:
                previous_steps_string = "_".join(["".join([st[0:3] for st in steps.split("_")]) for steps in
                                                  preprocessing_parameters.analyses_performed])
                # Checking if the dir for this specific signal already exists:
                signal_folder = Path(save_root, signal, previous_steps_string)
                if not os.path.isdir(signal_folder):
                    os.makedirs(signal_folder)
                # checking if the data were already generated
                if 'epochs' in locals() and signal in epochs:
                    # Now looping through each unique level of the condition we want to have one plot per level:
                    for cond_lvl in list(epochs[signal].metadata[step_parameters["plot_per_cond"]].unique()):
                        # Getting the data of only this conditions:
                        epochs_to_plot = epochs[signal].copy()['{0} == "{1}"'.format(
                            step_parameters["plot_per_cond"], cond_lvl)].load_data()
                        # Applying the baseline:
                        baseline_scaling(epochs_to_plot,
                                         correction_method=step_parameters[
                                             "baseline_meth_mapping"][signal])
                        # Selecting the channels of interest:
                        epochs_to_plot.pick_types(
                            **step_parameters["channel_types"])
                        # Looping through each channel to plot them:
                        for ch in epochs_to_plot.ch_names:
                            # Generate the full file name:
                            file_name, extension = \
                                os.path.splitext(
                                    file_name_generator(signal_folder, subject_info.files_prefix, ch + "_" + cond_lvl,
                                                        ".png",
                                                        data_type=preprocessing_parameters.data_type))
                            plot_ordered_epochs(epochs_to_plot, ch, within_condition=cond_lvl, file_prefix=file_name,
                                                save=True, subject=subject_info.SUBJ_ID, signal=signal,
                                                **step_parameters["plotting_parameters"])

                        # Generating summary plots:
                        # Generate the full file name:
                        file_name, extension = \
                            os.path.splitext(
                                file_name_generator(signal_folder, subject_info.files_prefix, cond_lvl,
                                                    ".png",
                                                    data_type=preprocessing_parameters.data_type))
                        plot_epochs_grids(epochs_to_plot, plot_type="raster", within_condition=cond_lvl,
                                          sub_conditions=None,
                                          file_prefix=file_name, save=True,
                                          signal=signal, units=None, subject=subject_info.SUBJ_ID,
                                          plot_standard_error=True, grid_dim=step_parameters["raster_grid_size"])
                        plot_epochs_grids(epochs_to_plot, plot_type="waveform", within_condition=cond_lvl,
                                          sub_conditions=None,
                                          file_prefix=file_name, save=True,
                                          signal=signal, units=None, subject=subject_info.SUBJ_ID,
                                          plot_standard_error=True, grid_dim=step_parameters["evoked_grid_size"])

        # --------------------------------------------------------------------------------------------------------------
        # Raise in error if the step that was passed isn't supported. This is to avoid the user assuming that something
        # was done when it wasn't.
        else:
            raise Exception(ERROR_UNKNOWN_STEP_TEXT.format(
                step=step, supported_steps=SUPPORTED_STEPS))

    end = time.time()
    print(end - start)
    print("DONE")


if __name__ == "__main__":
    preprocessing()
