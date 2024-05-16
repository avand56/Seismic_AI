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

# filtered_data = bandpass_filter(data, low=10, high=60, fs=200)  

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

# # Example usage
# # Assuming 'data' is your loaded 3D seismic dataset:
# processor = SeismicDataProcessor(data)

# # Compute amplitude and detect bright spots
# amplitude = processor.compute_amplitude()
# bright_spots = processor.detect_bright_spots(amplitude)

# # Compute phase and detect phase reversals
# phase = processor.compute_instantaneous_phase()
# phase_reversals = processor.detect_phase_reversals(phase)

# # Detect other features
# flat_spots = processor.detect_flat_spots()
# gas_chimneys = processor.detect_gas_chimneys()
# shadow_effects = processor.detect_shadow_effects()
