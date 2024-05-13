import segyio
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import hilbert, butter, filtfilt


# Apply a band-pass filter
def bandpass_filter(data, low, high, fs, order=5):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, data, axis=0)

# filtered_data = bandpass_filter(data, low=10, high=60, fs=200)  # Example values, adjust as necessary


# def detect_flat_spots(data, threshold):
#     # Calculate the horizontal gradient
#     gradient = np.abs(np.gradient(data, axis=1))
    
#     # Threshold the gradient to find areas with low change
#     flat_spots = gradient < threshold
    
#     return flat_spots

# flat_spots = detect_flat_spots(filtered_data, threshold=0.1)  # Adjust threshold based on data characteristics

# Visualizing flat spots
# plt.imshow(flat_spots, aspect='auto', cmap='gray')
# plt.title('Detected Flat Spots')
# plt.xlabel('Trace')
# plt.ylabel('Time/Sample Index')
# plt.show()


# Perform morphological closing to fill small holes in the detection
# closed_spots = scipy.ndimage.binary_closing(flat_spots, structure=np.ones((5,5)))

# plt.imshow(closed_spots, aspect='auto', cmap='gray')
# plt.title('Refined Flat Spots')
# plt.xlabel('Trace')
# plt.ylabel('Time/Sample Index')
# plt.show()



# def calculate_instantaneous_phase(data):
#     analytic_signal = hilbert(data, axis=0)  # Apply Hilbert transform along time axis
#     instantaneous_phase = np.unwrap(np.angle(analytic_signal), axis=0)
#     return instantaneous_phase

# Example use:
# data should be your seismic cube data
# instantaneous_phase = calculate_instantaneous_phase(data)

# Visualization
# plt.imshow(instantaneous_phase[:, :, some_slice], aspect='auto', cmap='twilight')
# plt.colorbar()
# plt.title('Instantaneous Phase')
# plt.xlabel('Trace')
# plt.ylabel('Time/Sample Index')
# plt.show()


# def detect_gas_chimneys(data, energy_threshold):
#     # Calculate energy or amplitude variations
#     energy = np.sum(data**2, axis=0)
    
#     # Detect regions with energy significantly different from surroundings
#     high_energy = energy > np.mean(energy) * energy_threshold
    
#     return high_energy

# # Example use:
# high_energy_regions = detect_gas_chimneys(data, energy_threshold=1.5)  # Adjust threshold as needed

# Visualization
# plt.imshow(high_energy_regions, aspect='auto', cmap='Reds')
# plt.title('Detected Gas Chimneys')
# plt.xlabel('Trace')
# plt.ylabel('Crossline Index')
# plt.show()

# def detect_shadow_effects(data, attenuation_threshold):
#     # Assuming data has been normalized or standard deviations are calculated
#     mean_amplitude = np.mean(data, axis=0)
#     std_deviation = np.std(data, axis=0)

#     # Identify areas with significantly lower amplitude than average
#     shadows = std_deviation < np.mean(std_deviation) * attenuation_threshold
    
#     return shadows

# Example use:
# shadow_effects = detect_shadow_effects(data, attenuation_threshold=0.8)  # Adjust threshold based on data

# Visualization
# plt.imshow(shadow_effects, aspect='auto', cmap='Blues')
# plt.title('Shadow Effects Below Gas Features')
# plt.xlabel('Trace')
# plt.ylabel('Time/Sample Index')
# plt.show()

def seismic_data_dhi(file_path):
    # Load seismic data
    file_path = 'path_to_seismic_data.segy'
    with segyio.open(file_path, "r", strict=False) as segyfile:
        segyfile.mmap()
        data = segyio.tools.cube(segyfile)
        sample_rate = segyio.tools.dt(segyfile) / 1000.0  # Sample rate in ms

    print("Data shape:", data.shape)


# Function to compute amplitude and detect bright spots
def compute_amplitude(data):
    return np.abs(data)

def detect_bright_spots(amplitude, threshold=1.3):
    mean_amp = np.mean(amplitude, axis=0)
    return amplitude > mean_amp * threshold

# Function to compute instantaneous phase and detect phase reversals
def compute_instantaneous_phase(data):
    analytic_signal = hilbert(data, axis=0)
    return np.unwrap(np.angle(analytic_signal), axis=0)

def detect_phase_reversals(phase):
    phase_diff = np.diff(phase, axis=0)
    return np.abs(phase_diff) > np.pi / 2  # Change in phase greater than 90 degrees

# Function to detect flat spots by examining consistency in amplitude over a horizontal line
def detect_flat_spots(data, window_size=5, threshold=0.1):
    horizontal_grad = np.abs(np.gradient(data, axis=2))
    smoothed = filtfilt(np.ones(window_size)/window_size, [1], horizontal_grad, axis=0)
    return smoothed < threshold

# Function to detect gas chimneys and shadow effects based on energy attenuation
def detect_gas_chimneys(data, energy_threshold=1.5):
    energy = np.sum(data**2, axis=0)
    return energy > np.mean(energy) * energy_threshold

def detect_shadow_effects(data, attenuation_threshold=0.5):
    std_deviation = np.std(data, axis=0)
    return std_deviation < np.mean(std_deviation) * attenuation_threshold


def calculate_thresholds(data, factor=1.3):
    """
    Calculate adaptive thresholds for feature detection based on data statistics.
    
    Args:
    data (numpy.ndarray): The seismic data array.
    factor (float): A multiplier to adjust sensitivity of detection.
    
    Returns:
    dict: Dictionary containing calculated thresholds for various features.
    """
    thresholds = {}
    mean_amp = np.mean(np.abs(data))
    std_amp = np.std(np.abs(data))
    
    # Bright spots are considered to be significantly above the mean amplitude
    thresholds['bright_spots'] = mean_amp + factor * std_amp
    
    # Phase reversals might occur where the change in phase is abrupt
    thresholds['phase_reversals'] = np.pi / 2  # 90 degrees threshold as a standard, can be adjusted
    
    # Flat spots detection could be sensitive to how flat the area is, here defined as low gradient
    thresholds['flat_spots'] = std_amp * factor * 0.1  # Very sensitive to gradients
    
    # Gas chimneys may exhibit significantly higher energy
    thresholds['gas_chimneys'] = np.mean(np.sum(data**2, axis=0)) * factor
    
    # Shadow effects are based on attenuation seen as reductions in standard deviation
    thresholds['shadow_effects'] = np.mean(np.std(data, axis=0)) * (1 - factor * 0.3)
    
    return thresholds

def process_seismic_data(data, thresholds):
    """
    Process each trace in each section of the seismic dataset.

    Args:
    data (numpy.ndarray): 3D numpy array of seismic data [inlines, samples, crosslines].
    thresholds (dict): Dictionary of thresholds for various feature detections.

    Returns:
    dict: A dictionary containing feature maps for bright spots, flat spots, gas chimneys, and shadow effects.
    """
    # Initialize result dictionaries
    results = {
        'bright_spots': np.zeros_like(data, dtype=bool),
        'flat_spots': np.zeros_like(data, dtype=bool),
        'gas_chimneys': np.zeros_like(data, dtype=bool),
        'shadow_effects': np.zeros_like(data, dtype=bool)
    }
    
    # Process each inline
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            trace_data = data[i, :, j]

            # Compute amplitude for bright spots detection
            amplitude = compute_amplitude(trace_data)
            results['bright_spots'][i, :, j] = detect_bright_spots(amplitude, threshold=thresholds['bright_spots'])

            # Compute instantaneous phase for flat spots (assuming method is adapted for individual traces)
            phase = compute_instantaneous_phase(trace_data)
            results['flat_spots'][i, :, j] = detect_flat_spots(trace_data, threshold=thresholds['flat_spots'])

            # Gas chimneys (energy-based approach)
            results['gas_chimneys'][i, :, j] = detect_gas_chimneys(trace_data, energy_threshold=thresholds['gas_chimneys'])

            # Shadow effects (attenuation-based)
            results['shadow_effects'][i, :, j] = detect_shadow_effects(trace_data, attenuation_threshold=thresholds['shadow_effects'])

    return results

# Example usage
thresholds = calculate_thresholds(data, factor=1.5)
features = process_seismic_data(data, thresholds)





# Example of using the function
thresholds = calculate_thresholds(inline_data, factor=1.5)

# Adjusting detection functions to use calculated thresholds
bright_spots = detect_bright_spots(amplitude, threshold=thresholds['bright_spots'])
phase_reversals = detect_phase_reversals(phase)  # Phase reversals use a fixed physical threshold
flat_spots = detect_flat_spots(inline_data, threshold=thresholds['flat_spots'])
gas_chimneys = detect_gas_chimneys(inline_data, energy_threshold=thresholds['gas_chimneys'])
shadow_effects = detect_shadow_effects(inline_data, attenuation_threshold=thresholds['shadow_effects'])


plt.figure(figsize=(15, 10))
plt.subplot(321)
plt.imshow(inline_data, aspect='auto')
plt.title('Original Data')

plt.subplot(322)
plt.imshow(bright_spots, aspect='auto')
plt.title('Bright Spots')

plt.subplot(323)
plt.imshow(phase_reversals, aspect='auto')
plt.title('Phase Reversals')

plt.subplot(324)
plt.imshow(flat_spots, aspect='auto')
plt.title('Flat Spots')

plt.subplot(325)
plt.imshow(gas_chimneys, aspect='auto')
plt.title('Gas Chimneys')

plt.subplot(326)
plt.imshow(shadow_effects, aspect='auto')
plt.title('Shadow Effects')

plt.tight_layout()
plt.show()
