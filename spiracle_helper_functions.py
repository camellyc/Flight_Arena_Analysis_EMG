import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import os

#=============================================
# 1. DATA READING AND PREPROCESSING FUNCTIONS
#=============================================

def load_and_reshape_data(file_path, duration=None, sampling_rate=20000):
    """
    Load and reshape data from .mat or binary file.
    
    Parameters:
    - file_path: Path to the data file
    - duration: Duration in seconds to load (None loads entire file)
    - sampling_rate: Sampling rate of the data
    
    Returns:
    - data: Reshaped data array (9 x time_points)
    """
    try:
        if file_path.endswith('.mat'):
            mat_data = loadmat(file_path)
            data_var = None
            for key in mat_data.keys():
                if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) == 2:
                    if mat_data[key].shape[0] == 9 or mat_data[key].shape[1] == 9:
                        data_var = key
                        break
            
            if data_var is None:
                raise ValueError("Could not find appropriate data array in .mat file")
                
            data = mat_data[data_var]
            
            if data.shape[0] != 9:
                data = data.T
                
            if duration is not None:
                samples = int(duration * sampling_rate)
                data = data[:, :samples]
                
        else:
            if duration is not None:
                num_samples = 9 * int(duration * sampling_rate)
                with open(file_path, 'rb') as fid:
                    data = np.fromfile(fid, dtype=np.float64, count=num_samples)
            else:
                data = np.memmap(file_path, dtype=np.float64, mode='r')
            
            data = data.reshape((-1, 9)).T
        
        print(f"Loaded data shape: {data.shape}")
        return data
        
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

def extract_traces(data):
    """
    Extract relevant traces from the data array.
    
    Returns:
    - wingbeat_freq: Data[3] containing wingbeat frequency
    - spike_raw: Data[5] containing EMG data
    - opto_stim: Data[6] containing optogenetic stimulus data
    - wing_beat_amplitude: Average of Data[1] and Data[2]
    - x_position: Data[4] containing x position
    """
    wing_beat_amplitude = (data[1] + data[2]) / 2  # Average of rows 1 and 2
    return data[3], data[5], data[6], wing_beat_amplitude, data[4], data[7]  # WBF, spike, opto, amplitude, x_pos, hutchens

def smooth_data(data, window_size=100):
    """
    Smooth data using a sliding window average.
    
    Parameters:
    - data: Input data array
    - window_size: Size of the smoothing window
    
    Returns:
    - Smoothed data array
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

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

def transform_opto_stimulus(data7, sampling_rate):
    """
    Transform optogenetic stimulus data into binary signal.
    
    Parameters:
    - data7: Raw optogenetic stimulus data
    - sampling_rate: Sampling rate in Hz
    
    Returns:
    - Binary stimulus signal (0 or 1)
    """
    transformed_data7 = np.zeros_like(data7)
    stim_on = False
    stim_start_idx = 0
    max_pulse_gap_samples = int(30 * sampling_rate / 1000)
    
    block_lengths = []
    block_count = 0
    last_stim_end_idx = -1

    for i in range(1, len(data7)):
        if data7[i] >= 0.9 and not stim_on:
            if stim_start_idx == 0:
                stim_start_idx = i
            stim_on = True
        elif data7[i] < 1 and stim_on:
            stim_on = False
            last_stim_end_idx = i

        if (stim_start_idx != 0 and (last_stim_end_idx != -1) and 
            (i - last_stim_end_idx > max_pulse_gap_samples)):
            if last_stim_end_idx > stim_start_idx:
                transformed_data7[stim_start_idx:last_stim_end_idx] = 1
                block_length = last_stim_end_idx - stim_start_idx
                block_lengths.append(block_length)
                block_count += 1
            stim_start_idx = 0

    if stim_start_idx != 0 and last_stim_end_idx > stim_start_idx:
        transformed_data7[stim_start_idx:last_stim_end_idx] = 1
        block_length = last_stim_end_idx - stim_start_idx
        block_lengths.append(block_length)
        block_count += 1
    
    block_lengths_in_seconds = [length / sampling_rate for length in block_lengths]
    print(f"Number of blocks detected: {block_count}")
    print(f"Lengths of detected blocks (in seconds): {block_lengths_in_seconds}")

    return transformed_data7

def generate_time_axis(data_length, sampling_rate):
    """
    Generate time axis based on data length and sampling rate.
    
    Parameters:
    - data_length: Number of samples
    - sampling_rate: Sampling rate in Hz
    
    Returns:
    - Time axis array in seconds
    """
    return np.arange(data_length) / sampling_rate

def detect_stim_onsets(transformed_opto_stim, sampling_rate):
    """
    Detect actual stimulus onset times from the opto signal.
    
    Parameters:
    - transformed_opto_stim: Binary stimulus signal
    - sampling_rate: Sampling rate in Hz
    
    Returns:
    - List of onset times in seconds
    """
    onsets = np.where(np.diff(transformed_opto_stim) > 0)[0]
    onset_times = onsets / sampling_rate
    return onset_times

def load_stim_orders(mat_file):
    """
    Load stimulus orders from .mat file.
    
    Parameters:
    - mat_file: Path to .mat file
    
    Returns:
    - List of stimulus durations for each session
    """
    try:
        mat_data = loadmat(mat_file)
        stim_orders = mat_data['allRandomizedStimOrders']
        
        # Convert to Python list format and flatten the structure
        parsed_orders = []
        for session in stim_orders:
            if isinstance(session[0], np.ndarray):
                # Extract each stimulus duration from the nested structure
                session_stims = [stim for stim in session[0][0]]
                parsed_orders.append(session_stims)
            else:
                parsed_orders.append(session.tolist())
                
        print(f"Loaded stimulus orders: {parsed_orders}")
        return parsed_orders
    except Exception as e:
        print(f"Error loading stimulus orders: {e}")
        return None

def match_stim_times(detected_onsets, stim_orders, session_duration=90):
    """
    Match detected stimulus onsets with planned stimulus durations and add 0ms stimuli.
    
    Parameters:
    - detected_onsets: List of detected onset times in seconds
    - stim_orders: List of planned stimulus durations for each session
    - session_duration: Duration of each session in seconds
    
    Returns:
    - List of tuples (onset_time, duration_ms) for all stimuli
    """
    all_stim_times = []
    detected_idx = 0
    current_time = 0
    
    for session_idx, session_stims in enumerate(stim_orders):
        n_stims = len(session_stims)
        interval = session_duration / (n_stims + 1)
        
        for stim_idx, stim_duration in enumerate(session_stims):
            planned_time = current_time + (stim_idx + 1) * interval
            
            if isinstance(stim_duration, (list, np.ndarray)):
                # If stim_duration is a list/array, extract the first non-zero value or use 0
                actual_duration = next((d for d in stim_duration if d != 0), 0)
            else:
                actual_duration = stim_duration
                
            if actual_duration == 0:
                # For 0ms stimuli, use the planned time
                all_stim_times.append((planned_time, 0))
            else:
                # For non-zero stimuli, use the nearest detected onset
                if detected_idx < len(detected_onsets):
                    # Find nearest detected onset to planned time
                    detected_time = detected_onsets[detected_idx]
                    if abs(detected_time - planned_time) < interval/2:  # Within half interval
                        all_stim_times.append((detected_time, actual_duration))
                        detected_idx += 1
                    else:
                        print(f"Warning: No matching detected onset for {actual_duration}ms stimulus at {planned_time}s")
                        all_stim_times.append((planned_time, actual_duration))
        
        current_time += session_duration
    
    return sorted(all_stim_times, key=lambda x: x[0])

def process_single_file(file_path, output_dir):
    """
    Process a single file and save individual bouts.
    """
    print(f"\nProcessing file: {file_path}")
    
    try:
        # Load and process data
        data = load_and_reshape_data(file_path)
        wingbeat_freq, spike_raw, opto_stim, wing_amp, x_pos, hutchens = extract_traces(data)
        
        # Process signals
        time_in_seconds = np.arange(len(wingbeat_freq)) / 20000
        wingbeat_freq_smoothed = smooth_data(wingbeat_freq) * 100
        wing_amp_smoothed = smooth_data(wing_amp)
        transformed_opto_stim = transform_opto_stimulus(opto_stim, 20000)
        _, spike_rate, spike_count, spike_amplitudes = process_spikes(spike_raw, 20000)
        
        # Create data dictionary
        data_dict = {
            'Time (s)': time_in_seconds,
            'Wingbeat Frequency': wingbeat_freq_smoothed,
            'Wing Beat Amplitude': wing_amp_smoothed,
            'X Position': x_pos,
            'Raw EMG': spike_raw,
            'Spike Rate': spike_rate,
            'Spike Count': spike_count,
            'Spike Amplitude': spike_amplitudes,
            'Opto Stimulus': transformed_opto_stim,
            'hutchens': hutchens
        }
        
        # Detect actual stimulus onsets
        detected_onsets = detect_stim_onsets(transformed_opto_stim, 20000)
        print(f"Detected {len(detected_onsets)} stimulus onsets at times: {detected_onsets}")
        
        # Load planned stimulus information
        stim_orders = load_stim_orders(file_path)
        if stim_orders is None:
            raise ValueError("Could not load stimulus orders")
        
        # Match detected onsets with planned stimuli
        stim_times = match_stim_times(detected_onsets, stim_orders)
        print(f"Matched {len(stim_times)} total stimuli (including 0ms)")
        
        saved_files = []
        
        # Process each stimulus
        for i, (onset_time, duration_ms) in enumerate(stim_times, 1):
            print(f"\nProcessing bout {i}:")
            print(f"Onset: {onset_time:.2f}s")
            print(f"Duration: {duration_ms}ms")
            
            duration_s = duration_ms / 1000.0
            bout_dict = extract_bout(data_dict, onset_time, duration_s)
            
            # Create corrected stimulus trace
            bout_samples = len(bout_dict['Time (s)'])
            corrected_stim = np.zeros(bout_samples)
            if duration_ms > 0:
                stim_start_idx = int(5 * 20000)  # 5s pre-stim
                stim_end_idx = stim_start_idx + int(duration_s * 20000)
                corrected_stim[stim_start_idx:stim_end_idx] = 1
            
            bout_dict['Opto Stimulus'] = corrected_stim
            
            # Save bout
            file_prefix = os.path.splitext(os.path.basename(file_path))[0]
            saved_file = save_bout_to_csv(bout_dict, output_dir, file_prefix, i)
            saved_files.append(saved_file)
            print(f"Saved bout {i} to {os.path.basename(saved_file)}")
        
        return saved_files
    
    except Exception as e:
        print(f"Error in process_single_file: {str(e)}")
        return []
    
def plot_entire_single_file(file_path, output_dir, fs=20000, make_plot=True):
    """
    Process an entire input file (no bout selection) and save a single CSV output.
    
    Args:
        file_path (str): path to the raw input file
        output_dir (str): directory to write outputs
        fs (int): sampling rate (default 20000)
        make_plot (bool): if True, also saves an overview plot PNG
    
    Returns:
        str: path to saved CSV (or None if failure)
    """
    print(f"\nProcessing ENTIRE file: {file_path}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load and process data
        data = load_and_reshape_data(file_path)
        wingbeat_freq, spike_raw, opto_stim, wing_amp, x_pos, hutchens = extract_traces(data)

        n = len(wingbeat_freq)
        time_in_seconds = np.arange(n) / float(fs)

        # Smooth / transform
        wingbeat_freq_smoothed = smooth_data(wingbeat_freq) * 100
        wing_amp_smoothed = smooth_data(wing_amp)
        transformed_opto_stim = transform_opto_stimulus(opto_stim, fs)

        # Spikes
        _, spike_rate, spike_count, spike_amplitudes = process_spikes(spike_raw, fs)

        # Build dataframe-friendly dict
        data_dict = {
            'Time (s)': time_in_seconds,
            'Wingbeat Frequency': wingbeat_freq_smoothed,
            'Wing Beat Amplitude': wing_amp_smoothed,
            'X Position': x_pos,
            'Raw EMG': spike_raw,
            'Spike Rate': spike_rate,
            'Spike Count': spike_count,
            'Spike Amplitude': spike_amplitudes,
            'Opto Stimulus': transformed_opto_stim,
            'hutchens': hutchens
        }

        # Convert to DataFrame with basic length checks
        lengths = {k: len(v) for k, v in data_dict.items()}
        L = max(lengths.values())
        bad = {k: (len(v), L) for k, v in data_dict.items() if len(v) != L}
        if bad:
            raise ValueError(f"Trace length mismatch across signals: {bad}")

        df = pd.DataFrame(data_dict)

        # Save CSV
        file_prefix = os.path.splitext(os.path.basename(file_path))[0]
        out_csv = os.path.join(output_dir, f"{file_prefix}_entire.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved entire-file CSV: {out_csv}")

        # Optional overview plot
        if make_plot:
            out_png = os.path.join(output_dir, f"{file_prefix}_entire_overview.png")

            fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
            axes[0].plot(df['Time (s)'], df['Wingbeat Frequency'])
            axes[0].set_ylabel("WBF")

            axes[1].plot(df['Time (s)'], df['Wing Beat Amplitude'])
            axes[1].set_ylabel("Amp")

            axes[2].plot(df['Time (s)'], df['Spike Rate'])
            axes[2].set_ylabel("Spike rate")

            axes[3].plot(df['Time (s)'], df['Opto Stimulus'])
            axes[3].set_ylabel("Opto")
            axes[3].set_xlabel("Time (s)")

            fig.tight_layout()
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            print(f"Saved overview plot: {out_png}")

        return out_csv

    except Exception as e:
        print(f"Error in plot_entire_single_file: {str(e)}")
        return None

#=============================================
# 2. SPIKE DETECTION AND ANALYSIS FUNCTIONS
#=============================================

def calculate_adaptive_threshold(signal, sampling_rate, window_duration_sec=0.5, multiplier=3):
    """
    Calculate adaptive threshold for spike detection.
    
    Parameters:
    - signal: Raw EMG signal
    - sampling_rate: Sampling rate in Hz
    - window_duration_sec: Duration of sliding window in seconds
    - multiplier: Threshold multiplier for std deviation
    
    Returns:
    - Array of threshold values for each time point
    """
    window_size = int(window_duration_sec * sampling_rate)
    threshold = np.zeros_like(signal)
    
    for i in tqdm(range(len(signal)), desc="Calculating Adaptive Threshold"):
        start_idx = max(0, i - window_size)
        end_idx = min(len(signal), i + window_size)
        threshold[i] = np.std(signal[start_idx:end_idx]) * multiplier
    
    return threshold

def calculate_derivative(signal):
    """
    Calculate the derivative of the signal.
    
    Parameters:
    - signal: Input signal array
    
    Returns:
    - Derivative of the signal
    """
    return np.diff(signal, prepend=signal[0])


'''
Old detect spikes code, placing the spike at the time point of maximum derivative. Used before 2024/12/06
def detect_spikes(signal, derivative, adaptive_threshold, sampling_rate, min_spike_distance_ms=10):
    """
    Detect spikes in the EMG signal and measure their amplitudes.
    
    Parameters:
    - signal: Raw EMG signal
    - derivative: Signal derivative
    - adaptive_threshold: Threshold array
    - sampling_rate: Sampling rate in Hz
    - min_spike_distance_ms: Minimum distance between spikes in ms
    
    Returns:
    - spike_count: Binary array marking spike occurrences
    - spike_amplitudes: Array of spike amplitudes (0 where no spike)
    """
    spike_count = np.zeros(len(signal))
    spike_amplitudes = np.zeros(len(signal))
    min_distance = int(min_spike_distance_ms * sampling_rate / 1000)
    last_spike_index = -min_distance
    post_window_samples = int(5 * sampling_rate / 1000)  # 5ms window for trough detection

    for i in tqdm(range(1, len(signal) - 1), desc="Detecting Spikes"):
        if (derivative[i] > derivative[i - 1] and 
            derivative[i] > derivative[i + 1] and
            abs(signal[i]) > adaptive_threshold[i] and 
            (i - last_spike_index) > min_distance):
            
            # Mark spike occurrence
            spike_count[i] = 1
            
            # Measure amplitude (peak to nearest trough within 5ms)
            end_idx = min(len(signal), i + post_window_samples)
            trough_idx = i + np.argmin(signal[i:end_idx])
            amplitude = signal[i] - signal[trough_idx]
            spike_amplitudes[i] = amplitude
            
            last_spike_index = i

    return spike_count, spike_amplitudes 
'''

def detect_spikes(signal, derivative, adaptive_threshold, sampling_rate, min_spike_distance_ms=10):
    """
    Detect spikes in the EMG signal and measure their amplitudes.
    The spike is marked at the absolute peak within ±2.5ms window around derivative peak.
    Used after 2024/12/06.
    
    Parameters:
    - signal: Raw EMG signal
    - derivative: Signal derivative
    - adaptive_threshold: Threshold array
    - sampling_rate: Sampling rate in Hz
    - min_spike_distance_ms: Minimum distance between spikes in ms
    
    Returns:
    - spike_count: Binary array marking spike occurrences
    - spike_amplitudes: Array of spike amplitudes (0 where no spike)
    """
    spike_count = np.zeros(len(signal))
    spike_amplitudes = np.zeros(len(signal))
    min_distance = int(min_spike_distance_ms * sampling_rate / 1000)
    window_samples = int(2.5 * sampling_rate / 1000)  # 2.5ms on each side
    post_window_samples = int(5 * sampling_rate / 1000)  # 5ms window for trough detection
    last_spike_index = -min_distance
    
    for i in tqdm(range(window_samples, len(signal) - window_samples), desc="Detecting Spikes"):
        if (derivative[i] > derivative[i - 1] and
            derivative[i] > derivative[i + 1] and
            abs(signal[i]) > adaptive_threshold[i] and
            (i - last_spike_index) > min_distance):
            
            # Find absolute peak within ±2.5ms window
            start_idx = max(0, i - window_samples)
            end_idx = min(len(signal), i + window_samples + 1)
            window = signal[start_idx:end_idx]
            peak_offset = np.argmax(abs(window))
            absolute_peak_idx = start_idx + peak_offset
            
            # Mark spike at absolute peak
            spike_count[absolute_peak_idx] = 1
            
            # Measure amplitude (peak to nearest trough within 5ms)
            trough_end_idx = min(len(signal), absolute_peak_idx + post_window_samples)
            trough_idx = absolute_peak_idx + np.argmin(signal[absolute_peak_idx:trough_end_idx])
            amplitude = signal[absolute_peak_idx] - signal[trough_idx]
            spike_amplitudes[absolute_peak_idx] = amplitude
            
            last_spike_index = absolute_peak_idx
            
    return spike_count, spike_amplitudes

def process_spikes(spike_raw, sampling_rate, target_spike_rate=7.0, duration=None):
    """
    Main function for spike detection and processing.

    Returns:
    - time_in_seconds: Time array
    - spike_rate: Smoothed spike rate
    - spike_count: Binary spike detection array
    - spike_amplitudes: Array of spike amplitudes
    """
    if duration is not None:
        num_samples = int(duration * sampling_rate)
        spike_raw = spike_raw[:num_samples]

    # Calculate adaptive threshold
    adaptive_threshold = calculate_adaptive_threshold(spike_raw, sampling_rate)

    # Calculate derivative
    derivative = calculate_derivative(spike_raw)

    # Detect spikes and get amplitudes
    spike_count, spike_amplitudes = detect_spikes(spike_raw, derivative, adaptive_threshold, sampling_rate)

    # Calculate spike rate
    spike_rate = calculate_spike_rate(spike_count, sampling_rate, target_spike_rate)

    # Generate time axis
    time_in_seconds = generate_time_axis(len(spike_raw), sampling_rate)

    return time_in_seconds, spike_rate, spike_count, spike_amplitudes

def calculate_spike_rate(spike_count, sampling_rate, target_spike_rate=7.0):
    """
    Calculate smooth spike rate using Gaussian convolution.

    Parameters:
    - spike_count: Binary spike detection array
    - sampling_rate: Sampling rate in Hz
    - target_spike_rate: Target rate for smoothing window

    Returns:
    - Smoothed spike rate in Hz
    """
    sigma = sampling_rate / target_spike_rate
    kernel_size = int(6 * sigma)
    x = np.linspace(-3 * sigma, 3 * sigma, kernel_size)

    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel /= np.sum(gaussian_kernel)

    spike_rate = np.convolve(spike_count, gaussian_kernel, mode='same') * sampling_rate
    return spike_rate

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


#=============================================
# 3. BOUT DETECTION AND SAVING FUNCTIONS
#=============================================
def extract_bout(data_dict, stim_onset, stim_duration, pre_stim_time=5, post_stim_time=15, sampling_rate=20000):
    """
    Extract a data bout around a stimulus.
    
    Parameters:
    - data_dict: Dictionary containing all data traces
    - stim_onset: Stimulus onset time in seconds
    - stim_duration: Stimulus duration in seconds
    - pre_stim_time: Time before stimulus to include (seconds)
    - post_stim_time: Time after stimulus to include (seconds)
    - sampling_rate: Sampling rate of the data
    
    Returns:
    - Dictionary containing the extracted bout data
    """
    start_idx = max(0, int((stim_onset - pre_stim_time) * sampling_rate))
    end_idx = min(
        len(data_dict['Time (s)']),
        int((stim_onset + post_stim_time) * sampling_rate)
    )
    
    bout_dict = {}
    for key in data_dict:
        bout_dict[key] = data_dict[key][start_idx:end_idx]
        
    # Adjust time to be relative to stim onset
    bout_dict['Time (s)'] = bout_dict['Time (s)'] - stim_onset
    bout_dict['Stim Duration (s)'] = stim_duration
    
    return bout_dict

def save_bout_to_csv(bout_dict, output_dir, file_prefix, bout_number):
    """
    Save bout data to CSV file.
    
    Parameters:
    - bout_dict: Dictionary containing bout data
    - output_dir: Directory to save CSV file
    - file_prefix: Prefix for output filename
    - bout_number: Number of current bout
    
    Returns:
    - Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    stim_duration_ms = int(bout_dict['Stim Duration (s)'] * 1000)
    output_file = os.path.join(
        output_dir, 
        f"{file_prefix}_bout{bout_number}_stim{stim_duration_ms}ms.csv"
    )
    
    df = pd.DataFrame(bout_dict)
    df.to_csv(output_file, index=False)
    return output_file

#=============================================
# 4. VISUALIZATION FUNCTIONS
#=============================================

def set_plot_style():
    """
    Opt-in matplotlib defaults for editable-text SVGs (Affinity/Illustrator safe).
    Call once near the top of a notebook. Idempotent.
    """
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300

def plot_single_bout(csv_file, stim_color='lightgreen', output_dir=None):
    """
    Plot data from a single bout CSV file.
    Shows smoothed wing beat amplitude without delta calculation.
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    # Calculate delta WBF
    baseline_mask = (df['Time (s)'] >= -2) & (df['Time (s)'] <= 0)
    baseline_wbf = df.loc[baseline_mask, 'Wingbeat Frequency'].mean()
    delta_wbf = df['Wingbeat Frequency'] - baseline_wbf
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[2, 2, 1])
    plt.subplots_adjust(hspace=0.4)
    
    # First plot: WBF on left axis
    ax1.plot(df['Time (s)'], delta_wbf, color='royalblue', label='ΔWBF')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('ΔWBF (Hz)', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim(-100, 30)
    
    # Add wing beat amplitude on separate axis
    ax1_amp = ax1.twinx()
    ax1_amp.plot(df['Time (s)'], df['Wing Beat Amplitude'], color='orange', label='WBA')
    ax1_amp.set_ylabel('Wing Beat Amplitude (a.u.)', color='orange')
    ax1_amp.tick_params(axis='y', labelcolor='orange')
    ax1_amp.set_ylim(0, 10)  # Fixed range for amplitude
    
    # Add spike rate on third y-axis
    ax1_spike = ax1.twinx()
    # Move the third axis spine to the left
    ax1_spike.spines["right"].set_position(("axes", 1.15))
    ax1_spike.plot(df['Time (s)'], df['Spike Rate'], color='mediumseagreen', label='Spike Rate')
    ax1_spike.set_ylabel('Spike Rate (Hz)', color='mediumseagreen')
    ax1_spike.tick_params(axis='y', labelcolor='mediumseagreen')
    ax1_spike.set_ylim(0, 15)
    
    # Add stimulus shading
    ymin, ymax = ax1.get_ylim()
    ax1.fill_between(df['Time (s)'], ymin, ymax,
                    where=df['Opto Stimulus'] > 0,
                    color=stim_color, alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_amp.get_legend_handles_labels()
    lines3, labels3 = ax1_spike.get_legend_handles_labels()
    ax1_spike.legend(lines1 + lines2 + lines3, 
                    labels1 + labels2 + labels3, 
                    loc='upper right')
    
    # Second plot: Raw EMG
    ax2.plot(df['Time (s)'], df['Raw EMG'], color='black', label='Raw EMG', linewidth=0.5)
    ax2.set_ylabel('EMG (V)')
    
    # Add detected spikes to raw EMG plot
    spike_times = df['Time (s)'][df['Spike Count'] == 1]
    spike_amplitudes = df['Raw EMG'][df['Spike Count'] == 1]
    ax2.scatter(spike_times, spike_amplitudes, color='red', s=20, alpha=0.6, label='Detected Spikes')
    ax2.legend()
    
    # Add stimulus shading to EMG plot
    ymin, ymax = ax2.get_ylim()
    ax2.fill_between(df['Time (s)'], ymin, ymax,
                    where=df['Opto Stimulus'] > 0,
                    color=stim_color, alpha=0.3)
    
    # Third plot: Raster
    spike_times = df['Time (s)'][df['Spike Count'] == 1]
    ax3.scatter(spike_times, [0] * len(spike_times), marker='|', color='black', s=100)
    ax3.set_ylabel('Spikes')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylim(-0.5, 0.5)
    
    # Add stimulus shading to raster
    ymin, ymax = ax3.get_ylim()
    ax3.fill_between(df['Time (s)'], ymin, ymax,
                    where=df['Opto Stimulus'] > 0,
                    color=stim_color, alpha=0.3)
    
    # Add title with file info
    filename = os.path.basename(csv_file)
    plt.suptitle(filename, y=0.95)
    
    # Save or show plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_analysis.svg"),
            format='svg', bbox_inches='tight')
    
    plt.show()

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

def plot_raster(spike_count_window, time_window, sampling_rate, xlimit):
    """
    Plot raster plot of spikes.
    
    Parameters:
    - spike_count_window: Binary spike detection array for window
    - time_window: Time array for window
    - sampling_rate: Sampling rate in Hz
    - xlimit: Tuple of (xmin, xmax) for plotting
    """
    spike_indices = np.where(spike_count_window == 1)[0]
    spike_times = time_window[spike_indices]

    if len(spike_times) == 0:
        print("No spikes detected in this time window.")
        return

    plt.figure(figsize=(15, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.9, color='black')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spike Events')
    plt.title('Raster Plot of Detected Spikes')
    
    if xlimit:
        plt.xlim(xlimit)
    plt.show()

def visualize_spike_in_time_window(time, raw_spike_data, spike_count, sampling_rate, 
                                 duration=None, pre_stim_time=0, block_start=0):
    """
    Plot raw spike data with detected spikes in specified window.
    
    Parameters:
    - time: Time array
    - raw_spike_data: Raw EMG data
    - spike_count: Binary spike detection array
    - sampling_rate: Sampling rate in Hz
    - duration: Optional duration limit
    - pre_stim_time: Time before stimulus
    - block_start: Start index of block
    """
    if duration is not None:
        num_samples = int(duration * sampling_rate)
        time = time[:num_samples]
        raw_spike_data = raw_spike_data[:num_samples]
        spike_count = spike_count[:num_samples]

    stim_on_time = time[block_start]
    start_idx = max(0, block_start - int(pre_stim_time * sampling_rate))
    end_idx = min(len(time), block_start + int(duration * sampling_rate))

    time_windowed = time[start_idx:end_idx] - stim_on_time
    raw_spike_windowed = raw_spike_data[start_idx:end_idx]
    spike_count_windowed = spike_count[start_idx:end_idx]

    plt.figure(figsize=(15, 5))
    plt.plot(time_windowed, raw_spike_windowed, label='Raw Spike Trace', color='black')

    detected_spike_times = time_windowed[spike_count_windowed == 1]
    detected_spike_amplitudes = raw_spike_windowed[spike_count_windowed == 1]
    plt.scatter(detected_spike_times, detected_spike_amplitudes, 
               color='green', label='Detected Spikes', s=50)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Raw Spike Amplitude (V)')
    plt.title(f'Raw Spike Trace with Detected Spikes (First {duration or "Full"} Seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_and_plot(stim_blocks, time, wingbeat_freq_smoothed, spike_rate, 
                    transformed_opto_stim, spike_count, sampling_rate, 
                    pre_stim_time, post_stim_time):
    """
    Analyze and plot data for each stimulus block.
    
    Parameters:
    - stim_blocks: List of stimulus block indices
    - time, wingbeat_freq_smoothed, spike_rate, transformed_opto_stim, spike_count: Data arrays
    - sampling_rate: Sampling rate in Hz
    - pre_stim_time, post_stim_time: Window boundaries relative to stimulus
    """
    for block_start, block_end in stim_blocks:
        # Calculate window indices
        stim_on_time = time[block_start]
        start_idx = max(0, block_start - int(pre_stim_time * sampling_rate))
        end_idx = min(len(time), block_start + int(post_stim_time * sampling_rate))

        # Extract windowed data
        time_window = time[start_idx:end_idx] - stim_on_time
        wingbeat_freq_window = wingbeat_freq_smoothed[start_idx:end_idx]
        spike_rate_window = spike_rate[start_idx:end_idx]
        transformed_stim_window = transformed_opto_stim[start_idx:end_idx]
        spike_count_window = spike_count[start_idx:end_idx]

        # Create plot
        fig, ax1 = plt.subplots(figsize=(15, 5))
        
        # Plot stimulus
        ax1.fill_between(time_window, 0, 250 * transformed_stim_window, 
                        color='lawngreen', alpha=0.1, label='Optogenetic Stimulus')
        
        # Plot WBF
        ax1.plot(time_window, wingbeat_freq_window, color='blue', label='Wingbeat Frequency (WBF)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Wingbeat Frequency (WBF)', color='blue')
        ax1.tick_params(axis='y', colors='blue')
        ax1.set_ylim(100, 260)
        
        # Plot spike rate
        ax2 = ax1.twinx()
        ax2.plot(time_window, spike_rate_window, label='Spike Rate', color='green')
        ax2.set_ylabel('Spike Rate (spikes/sec)', color='green')
        ax2.tick_params(axis='y', colors='green')
        ax2.set_ylim(0, 15)
        
        plt.title('Wingbeat Frequency, Spike Rate, and Optogenetic Stimulus')
        plt.tight_layout()
        plt.show()

        # Plot raster
        plot_raster(spike_count_window, time_window, sampling_rate, 
                   xlimit=(time_window[0], time_window[-1]))
        print(f"Number of spikes in this window: {np.sum(spike_count_window)}")

def calculate_mean_and_sem(data_list):
    """
    Calculate mean and standard error of mean for list of arrays.
    
    Parameters:
    - data_list: List of data arrays
    
    Returns:
    - mean: Mean array
    - sem: Standard error of mean array
    """
    data_stack = np.vstack(data_list)
    mean = np.mean(data_stack, axis=0)
    sem = np.std(data_stack, axis=0) / np.sqrt(data_stack.shape[0])
    return mean, sem

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