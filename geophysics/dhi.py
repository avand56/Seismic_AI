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


class DHIComputations:
    def __init__(self, data, window_size=5, factor=1.3):
        self.data = data
        self.window_size = window_size
        self.factor = factor
        self.thresholds = self.calculate_thresholds()

    def calculate_thresholds(self):
        mean_amp = np.mean(np.abs(self.data))
        std_amp = np.std(np.abs(self.data))

        return {
            'bright_spots': mean_amp + self.factor * std_amp,
            'phase_reversals': np.pi / 2,  # 90 degrees
            'flat_spots': std_amp * self.factor * 0.1,
            'gas_chimneys': np.mean(np.sum(self.data**2, axis=0)) * self.factor,
            'shadow_effects': np.mean(np.std(self.data, axis=0)) * (1 - self.factor * 0.3)
        }

    def compute_amplitude(self):
        return np.abs(self.data)

    def detect_bright_spots(self, amplitude):
        mean_amp = np.mean(amplitude, axis=0)
        return amplitude > mean_amp * self.thresholds['bright_spots']

    def compute_instantaneous_phase(self):
        analytic_signal = hilbert(self.data, axis=0)
        return np.unwrap(np.angle(analytic_signal), axis=0)

    def detect_phase_reversals(self, phase):
        phase_diff = np.diff(phase, axis=0)
        return np.abs(phase_diff) > self.thresholds['phase_reversals']

    def detect_flat_spots(self):
        horizontal_grad = np.abs(np.gradient(self.data, axis=2))
        smoothed = filtfilt(np.ones(self.window_size)/self.window_size, [1], horizontal_grad, axis=0)
        return smoothed < self.thresholds['flat_spots']

    def detect_gas_chimneys(self):
        energy = np.sum(self.data**2, axis=0)
        return energy > self.thresholds['gas_chimneys']

    def detect_shadow_effects(self):
        std_deviation = np.std(self.data, axis=0)
        return std_deviation < self.thresholds['shadow_effects']

# Example usage
# Assuming 'data' is your loaded 3D seismic dataset:
processor = SeismicDataProcessor(data)

# Compute amplitude and detect bright spots
amplitude = processor.compute_amplitude()
bright_spots = processor.detect_bright_spots(amplitude)

# Compute phase and detect phase reversals
phase = processor.compute_instantaneous_phase()
phase_reversals = processor.detect_phase_reversals(phase)

# Detect other features
flat_spots = processor.detect_flat_spots()
gas_chimneys = processor.detect_gas_chimneys()
shadow_effects = processor.detect_shadow_effects()

# def calculate_thresholds(data, factor=1.3):
#     """
#     Calculate adaptive thresholds for feature detection based on data statistics.
    
#     Args:
#     data (numpy.ndarray): The seismic data array.
#     factor (float): A multiplier to adjust sensitivity of detection.
    
#     Returns:
#     dict: Dictionary containing calculated thresholds for various features.
#     """
#     thresholds = {}
#     mean_amp = np.mean(np.abs(data))
#     std_amp = np.std(np.abs(data))
    
#     # Bright spots are considered to be significantly above the mean amplitude
#     thresholds['bright_spots'] = mean_amp + factor * std_amp
    
#     # Phase reversals might occur where the change in phase is abrupt
#     thresholds['phase_reversals'] = np.pi / 2  # 90 degrees threshold as a standard, can be adjusted
    
#     # Flat spots detection could be sensitive to how flat the area is, here defined as low gradient
#     thresholds['flat_spots'] = std_amp * factor * 0.1  # Very sensitive to gradients
    
#     # Gas chimneys may exhibit significantly higher energy
#     thresholds['gas_chimneys'] = np.mean(np.sum(data**2, axis=0)) * factor
    
#     # Shadow effects are based on attenuation seen as reductions in standard deviation
#     thresholds['shadow_effects'] = np.mean(np.std(data, axis=0)) * (1 - factor * 0.3)
    
#     return thresholds

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
        'shadow_effects': np.zeros_like(data, dtype=bool),
        'phase_reversals': np.zeros_like(data[:, :-1, :], dtype=bool)  # Adjusted for the diff operation in phase reversals
    }
    
    # Process each inline
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            trace_data = data[i, :, j]

            # Compute amplitude for bright spots detection
            amplitude = compute_amplitude(trace_data)
            results['bright_spots'][i, :, j] = detect_bright_spots(amplitude, thresholds['bright_spots'])

            # Compute instantaneous phase for phase reversals
            phase = compute_instantaneous_phase(trace_data)
            results['phase_reversals'][i, :, j] = detect_phase_reversals(phase, thresholds['phase_reversals'])

            # Detect flat spots using the original seismic amplitude data
            results['flat_spots'][i, :, j] = detect_flat_spots(trace_data, 5, thresholds['flat_spots'])

            # Gas chimneys (energy-based approach)
            results['gas_chimneys'][i, :, j] = detect_gas_chimneys(trace_data, thresholds['gas_chimneys'])

            # Shadow effects (attenuation-based)
            results['shadow_effects'][i, :, j] = detect_shadow_effects(trace_data, thresholds['shadow_effects'])

    return results


# # Example usage
# thresholds = calculate_thresholds(data, factor=1.5)
# features = process_seismic_data(data, thresholds)





# # Example of using the function
# thresholds = calculate_thresholds(inline_data, factor=1.5)

# # Adjusting detection functions to use calculated thresholds
# bright_spots = detect_bright_spots(amplitude, threshold=thresholds['bright_spots'])
# phase_reversals = detect_phase_reversals(phase)  # Phase reversals use a fixed physical threshold
# flat_spots = detect_flat_spots(inline_data, threshold=thresholds['flat_spots'])
# gas_chimneys = detect_gas_chimneys(inline_data, energy_threshold=thresholds['gas_chimneys'])
# shadow_effects = detect_shadow_effects(inline_data, attenuation_threshold=thresholds['shadow_effects'])


# plt.figure(figsize=(15, 10))
# plt.subplot(321)
# plt.imshow(inline_data, aspect='auto')
# plt.title('Original Data')

# plt.subplot(322)
# plt.imshow(bright_spots, aspect='auto')
# plt.title('Bright Spots')

# plt.subplot(323)
# plt.imshow(phase_reversals, aspect='auto')
# plt.title('Phase Reversals')

# plt.subplot(324)
# plt.imshow(flat_spots, aspect='auto')
# plt.title('Flat Spots')

# plt.subplot(325)
# plt.imshow(gas_chimneys, aspect='auto')
# plt.title('Gas Chimneys')

# plt.subplot(326)
# plt.imshow(shadow_effects, aspect='auto')
# plt.title('Shadow Effects')

# plt.tight_layout()
# plt.show()
