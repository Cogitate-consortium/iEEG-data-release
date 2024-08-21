"""
This module provides functions for visualizing intracranial EEG (iEEG) data, including plotting power spectral
densities, bad channels, and custom iEEG images with logistic colormapping.

The functions in this module are designed to create visual representations of iEEG data to assist with analysis
and interpretation.

Author:
-------
Alex Lepauvre
Katarina Bendtz
Simon Henin

License:
--------
This code is licensed under the MIT License.
"""

from pathlib import Path

import matplotlib
import mne
import numpy as np
from matplotlib import pyplot as plt, colormaps, colors as mcolors
from mne.datasets import fetch_fsaverage
from mne.viz import plot_alignment, snapshot_brain_montage

from cog_ieeg.localization import add_fiducials
from cog_ieeg.utils import path_generator, save_param


def plot_channels_psd(raw, save_root, step, signal, file_prefix,
                      file_extension="-psd.png", plot_single_channels=False, channels_type=None):
    """
    Plot and save the power spectral density (PSD) of chosen electrodes.

    :param raw: mne raw object
    :param save_root: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    :param plot_single_channels: boolean, whether to plot single channels or only all of them superimposed
    :param channels_type: dict, list of the channels of interest
    :return: None
    """
    # Getting  the relevant channels:
    if channels_type is None:
        channels_type = {"ecog": True, "seeg": True}
    picks = mne.pick_types(raw.info, **channels_type)

    # Generate the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)
    # Create the name of the file:
    full_file_name = Path(save_path, '{}_desc-{}_ieeg{}'.format(file_prefix, step, file_extension))

    # ==========================================================
    # Plotting the psd from all the channels:
    raw.plot_psd(picks=picks, show=False)
    # Saving the figure:
    plt.savefig(full_file_name, dpi=300, transparent=True)
    plt.close()

    # ==========================================================
    # For all channels separately:
    if plot_single_channels:
        for pick in picks:
            raw.plot_psd(picks=[pick], show=False)
            full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}{}'.format(file_prefix, step,
                                                                           raw.ch_names[pick], file_extension))
            plt.savefig(full_file_name, dpi=300, transparent=True)
            plt.close()

    return None


def plot_bad_channels(raw, save_root, step, signal, file_prefix,
                      file_extension="bads.png", plot_single_channels=False, picks="bads"):
    """
    Plot the bad channels PSD and raw signal to show what is being discarded.

    :param raw: mne raw object
    :param save_root: string or path object, path to where the data should be saved
    :param step: string, name of the preprocessing step
    :param signal: string, name of the signal being saved
    :param file_prefix: string, prefix of the file to save
    :param file_extension: string, file name extension
    :param plot_single_channels: boolean, whether to plot single channels or only all of them superimposed
    :param picks: list, list of the channels of interest
    :return: None
    """
    # Handling picks input:
    if picks == "bads":
        picks = raw.info["bads"]

    # Generate the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)
    # Create the name of the file:
    full_file_name = Path(save_path, '{}_desc-{}_ieeg{}'.format(file_prefix, step, file_extension))

    if len(picks) > 0:
        # Plotting the psd from all the channels:
        fig, axs = plt.subplots(1, 1)
        plt.suptitle("Bad channels: N= " + str(len(picks))
                     + " out of " + str(len(raw.info["ch_names"])))
        # Plotting the average of the good channels with standard error for reference. Downsampling, otherwise too many
        # data points:
        good_channels_data, times = \
            raw.copy().resample(100).get_data(picks=mne.pick_types(
                raw.info, ecog=True, seeg=True), return_times=True)
        mean_good_data = np.mean(good_channels_data.T, axis=1)
        ste_good_data = np.std(good_channels_data.T, axis=1) / \
                        np.sqrt(good_channels_data.T.shape[1])
        axs.plot(times, mean_good_data, alpha=0.5, color="black")
        axs.fill_between(times, mean_good_data - ste_good_data, mean_good_data + ste_good_data, alpha=0.2,
                         color="black")
        # Plotting the time series of all the bad channels:
        axs.plot(times, raw.copy().resample(100).get_data(
            picks=picks).T, alpha=0.8, label=picks)
        # Adding labels and legend:
        axs.set_xlabel("time (s)")
        axs.set_ylabel("amplitude")
        axs.legend()
        # Plotting the psd
        # raw.plot_psd(picks=picks, show=False, ax=axs[1:2])
        # Saving the figure:
        plt.savefig(full_file_name, transparent=True)
        plt.close()
        # For all channels separately:
        if plot_single_channels:
            # Looping through each channels:
            for pick in picks:
                # Plotting the psd from this channel:
                fig, axs = plt.subplots(2, 1)
                axs[0].plot(times, mean_good_data, alpha=0.5, color="black")
                axs[0].fill_between(times, mean_good_data - ste_good_data, mean_good_data + ste_good_data,
                                    alpha=0.2,
                                    color="black")
                axs[0].plot(times, raw.copy().resample(
                    200).get_data(picks=pick).T, label=pick)
                axs[0].set_xlabel("time (s)")
                axs[0].set_ylabel("amplitude")
                # Adding labels and legend:
                axs[0].set_xlabel("time (s)")
                axs[0].set_ylabel("amplitude")
                axs[0].legend()
                # Plotting the psd
                raw.plot_psd(picks=pick, show=False, ax=axs[1])
                full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}{}'.format(file_prefix, step,
                                                                               raw.ch_names[pick], file_extension))
                plt.savefig(full_file_name, transparent=True)
                plt.close()

    return None


def plot_electrode_localization(mne_object, subject, fs_dir, param, save_root, step, signal, file_prefix,
                                file_extension='-loc.png', channels_to_plot=None,
                                plot_elec_name=False):
    """
    Plot and save the electrode localization.

    :param mne_object: mne object (raw, epochs, evoked...), contains the mne object with the channels info
    :param subject: string, subject ID
    :param fs_dir: string or path object, freesurfer directory containing the subject's data
    :param param: dict, contains analysis parameters
    :param save_root: string, directory to save the figures
    :param step: string, name of the analysis step for saving the parameters
    :param signal: string, name of the signal
    :param file_prefix: string, prefix for file saving
    :param file_extension: string, ending of the file name
    :param channels_to_plot: list, contains the different channels to plot
    :param plot_elec_name: string, whether or not to print the electrodes names onto the snapshot
    :return: None
    """
    if channels_to_plot is None:
        channels_to_plot = ["ecog", "seeg"]
    if mne_object.get_montage().get_positions()['coord_frame'] == "mni_tal":
        subject = "fsaverage"
        fetch_fsaverage(subjects_dir=fs_dir, verbose=True)
    # First, generating the root path to save the data:
    save_path = Path(save_root, step, signal)
    path_generator(save_path)

    # Adding the estimated fiducials and compute the transformation to head:
    montage, trans = add_fiducials(mne_object.get_montage(), fs_dir, subject)
    mne_object.set_montage(montage, on_missing="warn")

    brain_snapshot_files = []
    # Setting the two views
    snapshot_orientations = {
        "left": {"azimuth": 180, "elevation": None},
        "front": {"azimuth": 90, "elevation": None},
        "right": {"azimuth": 0, "elevation": None},
        "back": {"azimuth": -90, "elevation": None},
        "top": {"azimuth": 0, "elevation": 180},
        "bottom": {"azimuth": 0, "elevation": -180}
    }
    # We want to plot the seeg and ecog channels separately:
    for ch_type in channels_to_plot:
        try:
            data_to_plot = mne_object.copy().pick(ch_type)
        except ValueError:
            print("No {0} channels for this subject".format(ch_type))
            continue
        # Plotting the brain surface with the electrodes and making a snapshot
        if ch_type == "ecog":
            fig = plot_alignment(data_to_plot.info, subject=subject, subjects_dir=fs_dir,
                                 surfaces=['pial'], coord_frame='head', trans=trans)
        else:
            fig = plot_alignment(data_to_plot.info, subject=subject, subjects_dir=fs_dir,
                                 surfaces=['white'], coord_frame='head', trans=trans)

        for ori in snapshot_orientations.keys():
            if plot_elec_name:
                # Generating the full file name:
                full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}_view-{}_names{}'.format(file_prefix,
                                                                                             step,
                                                                                             ori, ch_type,
                                                                                             file_extension))
            else:
                full_file_name = Path(save_path, '{}_desc-{}_ieeg-{}_view-{}{}'.format(file_prefix, step, ori, ch_type,
                                                                                       file_extension))
            mne.viz.set_3d_view(fig, **snapshot_orientations[ori])
            xy, im = snapshot_brain_montage(
                fig, data_to_plot.info, hide_sensors=False)
            fig_2, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(im)
            ax.set_axis_off()
            if plot_elec_name is True:
                for ch in list(xy.keys()):
                    ax.annotate(ch, xy[ch], fontsize=14, color="white",
                                xytext=(-5, -5), xycoords='data', textcoords='offset points')
            plt.savefig(full_file_name, transparent=True)
            plt.close()
            brain_snapshot_files.append(full_file_name)
        mne.viz.close_3d_figure(fig)
    # Saving the config:
    save_param(param, save_path, step, signal, file_prefix, file_extension.split('.')[0])

    return None


def count_colors(values, cmap):
    """
    Count the number of entries for each value and map them to colors.

    :param values: list of values
    :param cmap: string, name of the colormap
    :return: dict, mapping keys to RGB colors
    """
    from collections import Counter
    # Count the number of entries for each values:
    counts = dict(Counter(values))

    # Get the list of counts from the dictionary
    counts_val = list(counts.values())

    # Create a colormap
    cmap = plt.cm.get_cmap(cmap)

    # Normalize the counts to [0, 1] to match the colormap's scale
    norm_counts = [(count - min(counts_val)) / (max(counts_val) - min(counts_val)) for count in counts_val]

    # Map normalized counts to colors
    colors = [cmap(norm) for norm in norm_counts]

    # Convert RGBA colors to RGB
    rgb_colors = [(r, g, b) for r, g, b, _ in colors]

    # Create a new dictionary mapping keys to RGB colors
    val_colors = dict(zip(counts.keys(), rgb_colors))

    return val_colors


def get_cmap_rgb_values(values, cmap=None, center=None):
    """
    Get RGB values for a list of values mapping onto a specified color bar. If a midpoint is set,
    the color bar will be normalized accordingly.

    :param values: list of floats, list of values for which to obtain a color map
    :param cmap: string, name of the colormap
    :param center: float, value on which to center the colormap
    :return: list of rgb triplets, color for each passed value
    """
    if cmap is None:
        cmap = "RdYlBu_r"
    if center is None:
        center = np.mean([min(values), max(values)])
    # Create the normalization function:
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min(values), vcenter=center, vmax=max(values))
    colormap = colormaps.get_cmap(cmap)
    colors = [colormap(norm(value)) for value in values]

    return colors


def plot_ieeg_image(epo, channel, order=None, show=False, units="HGP (norm.)", scalings=1, cmap="RdYlBu_r",
                    center=1, ylim_prctile=95, logistic_cmap=True, ci=0.95, evk_method="mean", ci_method="mean",
                    evk_colors="k", vlines_s=0):
    """
    Plot an iEEG (intracranial EEG) image with the option to use logistic normalization for the colormap,
    including confidence intervals and evoked responses.

    Parameters:
    -----------
    epo : mne.Epochs
        The MNE Epochs object containing the data to be plotted.
    channel : str
        The name of the channel to plot.
    order : array-like | None
        If not None, reorder the images by this array. Default is None.
    show : bool, optional
        Whether to display the figure immediately. Default is False.
    units : str, optional
        The unit label for the y-axis. Default is "HGP (norm.)".
    scalings : float, optional
        Scaling factor for the data. Default is 1.
    cmap : str or None, optional
        Colormap for the image. Default is "RdYlBu_r". If None, uses the default colormap.
    center : float or None, optional
        Center value for the color normalization (used for symmetric data). Default is 1.
        If None, no centering is applied. Is overriden by logistic_cmap if activated
    ylim_prctile : float or list of floats, optional
        Percentile(s) to define the y-axis limits. Default is 95.
        If an integer is provided, it defines both upper and lower limits symmetrically.
    logistic_cmap : bool, optional
        Whether to apply logistic mapping to the colormap to emphasize the data's dynamic range.
        Default is True.
    ci : float, optional
        Confidence interval percentage. Default is 0.95 (95% confidence interval).
    evk_method : str, optional
        Method to compute the evoked response. Default is "mean".
    ci_method : str, optional
        Method to compute the confidence interval. Default is "mean".
    evk_colors : str, optional
        Color for the evoked response plot. Default is "k" (black).

    Returns:
    --------
    figs : list
        A list of figures generated by the plot.
    """

    # Define the logistic forward and inverse functions
    def _forward(x, x0=0, k=1):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def _inverse(y, x0=0, k=1):
        return x0 - (1 / k) * np.log((1 / y) - 1)

    # Create the colormap:
    if cmap is None:
        cmap = plt.get_cmap("RdYlBu_r")
    else:
        cmap = plt.get_cmap(cmap)

    # Get the y limits:
    data = np.squeeze(epo.get_data(picks=channel))
    if ylim_prctile is None:
        vmax = np.max(data)
        vmin = np.min(data)
    else:
        if isinstance(ylim_prctile, int):
            ylim_prctile = [ylim_prctile, 100 - ylim_prctile]

        vmax = np.percentile(data, ylim_prctile[0])
        vmin = np.percentile(data, ylim_prctile[1])

    # Apply centered normalization if `center` is provided
    if center is not None:
        norm = matplotlib.colors.CenteredNorm(vcenter=center)
        cmap = mcolors.ListedColormap(cmap(norm(np.linspace(vmin, vmax, 256))))

    # Apply logistic normalization if enabled
    if logistic_cmap:
        norm = matplotlib.colors.FuncNorm((lambda x: _forward(x, x0=0, k=1),
                                           lambda y: _inverse(y, x0=0, k=1)),
                                          vmin=vmin, vmax=vmax)
        cmap = mcolors.ListedColormap(cmap(norm(np.linspace(vmin, vmax, 256))))
    if vlines_s is not None:
        if isinstance(vlines_s, int) or isinstance(vlines_s, float):
            vlines_s = [vlines_s]
    # Plot the image:
    figs = mne.viz.plot_epochs_image(epo, picks=channel, show=show, order=order,
                                     units=dict(ecog=units, seeg=units),
                                     scalings=dict(ecog=scalings, seeg=scalings),
                                     evoked=True, cmap=cmap, vmin=vmin, vmax=vmax)

    # Compute the evoked response:
    evk = np.squeeze(epo.average(picks=channel, method=evk_method).get_data())

    # Compute the confidence interval:
    ci_low, ci_up = mne.stats.bootstrap_confidence_interval(epo.get_data(picks=channel),
                                                            stat_fun=ci_method, ci=ci)

    # Clear the existing axis and replace it:
    ax = figs[0].axes[1]
    ax.cla()
    ax.plot(epo.times, evk, color=evk_colors)
    ax.fill_between(epo.times, np.squeeze(ci_up), np.squeeze(ci_low), color=evk_colors, alpha=0.3)
    if vlines_s is not None:
        for x in vlines_s:
            ax.axvline(x=x, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle='dotted')
    ax.set_xlim([epo.times[0], epo.times[-1]])
    ax.set_ylabel(units)
    ax.set_xlabel("Time (s)")

    return figs
