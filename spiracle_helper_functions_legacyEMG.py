"""
Legacy EMG helper functions.

These functions were part of earlier Flight Arena EMG analysis explorations but are
no longer called by any of the current top-level pipeline notebooks
(Step1-Step6, mat_to_summary_noEMG_step1, convert_binary_to_mat).

They are kept here for historical reference and to support the old notebooks in the
`legacy/` folder (e.g. legacy/EMG_analysis.ipynb,
legacy/WBF_spikerate_spikeraster_opto.ipynb), which still reference them.

To use these in a notebook, `%run` this file in addition to
spiracle_helper_functions.py, or import the specific names you need.

Moved out of spiracle_helper_functions.py on 2026-06-02.
"""

import numpy as np
import matplotlib.pyplot as plt

# process_spikes_absolute_threshold depends on these helpers, which remain in the
# active module.
from spiracle_helper_functions import calculate_spike_rate, generate_time_axis


def normalize_data(data, new_min, new_max, old_min=None, old_max=None):
    """
    Normalize data to a new range.

    Parameters:
    - data: Input data array
    - new_min, new_max: Target range
    - old_min, old_max: Source range (if None, uses data min/max)

    Returns:
    - Normalized data array
    """
    if old_min is None:
        old_min = np.min(data)
    if old_max is None:
        old_max = np.max(data)
    return (data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


def process_spikes_absolute_threshold(spike_raw, sampling_rate, min_threshold, max_threshold,
                                    refractory_period_ms=2, duration=None):
    """
    Alternative spike detection using absolute thresholds.

    Parameters:
    - spike_raw: Raw EMG signal
    - sampling_rate: Sampling rate in Hz
    - min_threshold: Minimum voltage threshold
    - max_threshold: Maximum voltage threshold
    - refractory_period_ms: Minimum time between spikes
    - duration: Optional duration limit in seconds

    Returns:
    - time_in_seconds: Time array
    - spike_rate: Smoothed spike rate
    - spike_count: Binary spike detection array
    """
    if duration is not None:
        num_samples = int(duration * sampling_rate)
        spike_raw = spike_raw[:num_samples]

    refractory_period_samples = int(refractory_period_ms * sampling_rate / 1000)
    potential_spikes = ((np.abs(spike_raw) >= min_threshold) &
                       (np.abs(spike_raw) <= max_threshold)).astype(int)

    spike_count = np.zeros_like(potential_spikes)
    last_spike_idx = -refractory_period_samples

    for i in range(len(potential_spikes)):
        if potential_spikes[i] == 1 and i - last_spike_idx >= refractory_period_samples:
            spike_count[i] = 1
            last_spike_idx = i

    spike_rate = calculate_spike_rate(spike_count, sampling_rate)
    time_in_seconds = generate_time_axis(len(spike_raw), sampling_rate)

    return time_in_seconds, spike_rate, spike_count


def plot_data(time, data4_normalized, spike_rate, transformed_data7):
    """
    Plot main overview of all data traces.

    Parameters:
    - time: Time array in seconds
    - data4_normalized: Normalized wingbeat frequency
    - spike_rate: Processed spike rate
    - transformed_data7: Binary stimulus signal
    """
    fig, ax1 = plt.subplots(figsize=(18, 6), facecolor='k')
    ax1.set_facecolor('k')

    # Plot wingbeat frequency
    ax1.plot(time, data4_normalized, color='blue', linewidth=1.5, label='Wingbeat Frequency')
    ax1.set_xlabel('Time (seconds)', color='w')
    ax1.set_ylabel('Wingbeat Frequency (Hz)', color='w')

    # Set y-limits for wingbeat frequency
    min_data4 = np.min(data4_normalized)
    max_data4 = np.max(data4_normalized)
    y_range_data4 = max_data4 - min_data4
    ax1.set_ylim(min_data4 - 0.1 * y_range_data4, max_data4 + 0.1 * y_range_data4)

    # Style primary axis
    ax1.tick_params(axis='both', colors='w')
    for spine in ax1.spines.values():
        spine.set_color('w')

    # Create secondary axis for spike rate
    ax2 = ax1.twinx()
    ax2.plot(time, spike_rate, color='gray', alpha=0.5, linewidth=1.5, label='Spike Rate')
    ax2.set_ylabel('Spike Rate (Hz)', color='w')

    # Set y-limits for spike rate
    min_spike_rate = np.min(spike_rate)
    max_spike_rate = np.max(spike_rate)
    y_range_spike_rate = max_spike_rate - min_spike_rate
    ax2.set_ylim(min_spike_rate - 0.1 * y_range_spike_rate, max_spike_rate + 0.1 * y_range_spike_rate)
    ax2.tick_params(axis='y', colors='w')

    # Plot stimulus
    ax1.fill_between(time, 0, 20 * transformed_data7, color='lawngreen', alpha=0.5, label='Optogenetic Stimulus')

    plt.title('Normalized Data Plots', color='w')
    fig.legend(loc='center left', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, facecolor='k', labelcolor='w')
    plt.show()


def plot_spike_train(time_in_seconds, data6, spike_count, transformed_data7, xlimit):
    """
    Plot raw spike data with detected spikes and stimulus.

    Parameters:
    - time_in_seconds: Time array
    - data6: Raw EMG data
    - spike_count: Binary spike detection array
    - transformed_data7: Binary stimulus signal
    - xlimit: Tuple of (xmin, xmax) for plotting
    """
    fig, ax1 = plt.subplots(1, figsize=(15, 5))

    ax1.plot(time_in_seconds, data6, label='Voltage Data', color='black')
    ax1.scatter(time_in_seconds[np.where(spike_count == 1)[0]],
                data6[np.where(spike_count == 1)[0]],
                color='green', label='Detected Spikes', s=30)

    ax1.fill_between(time_in_seconds, 0, transformed_data7,
                     color='lawngreen', alpha=0.5,
                     label='Optogenetic Stimulus', zorder=0)

    ax1.set_title('Voltage Data with Detected Spikes and Optogenetic Stimulus')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_xlim(xlimit)
    ax1.legend()
    plt.tight_layout()
    plt.show()


def plot_mean_with_sem(time_window, mean_data, sem_data, color, label, ylabel, n_blocks):
    """
    Plot mean with standard error of mean shading.

    Parameters:
    - time_window: Time array
    - mean_data, sem_data: Data arrays
    - color: Plot color
    - label, ylabel: Plot labels
    - n_blocks: Number of blocks for label
    """
    plt.plot(time_window, mean_data, color=color, label=f'{label} (n={n_blocks})')
    plt.fill_between(time_window, mean_data - sem_data, mean_data + sem_data,
                    color=color, alpha=0.2)
    plt.ylabel(ylabel)
