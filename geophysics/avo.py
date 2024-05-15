import segyio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def read_seg_avo(file_path):

    with segyio.open(file_path, "r") as segyfile:
        # Memory-map the file to avoid loading it all at once
        segyfile.mmap()
        
        # Gather data dimensions from the SEG-Y file
        n_traces = segyfile.tracecount
        sample_count = segyfile.samples.size
        trace_headers = segyfile.attributes(segyio.TraceField.Offset)[:]
        
        # Read data into a numpy array
        data = segyio.tools.cube(segyfile)
        offsets = np.array(list(set(trace_headers)))
        offsets.sort()

    return print(f"Data loaded with dimensions: {data.shape} and offsets: {offsets}")


# Let's assume 'data' is a 3D numpy array of shape [traces, samples, offsets]
# and 'offsets' is a 1D numpy array containing the offset distances corresponding to the third dimension of 'data'

def calculate_avo_attributes(data, offsets):
    num_traces, num_samples, num_offsets = data.shape
    # Initialize arrays to store the AVO attributes
    gradients = np.zeros((num_traces, num_samples))
    intercepts = np.zeros((num_traces, num_samples))

    # Model to fit
    model = LinearRegression()

    # Loop over every trace and sample depth
    for i in range(num_traces):
        for j in range(num_samples):
            amplitudes = data[i, j, :]  # Amplitude vs. offset for this trace and depth
            # Reshape for sklearn
            offsets_reshaped = offsets.reshape(-1, 1)
            amplitudes_reshaped = amplitudes.reshape(-1, 1)

            # Fit linear regression model
            model.fit(offsets_reshaped, amplitudes_reshaped)

            # Store the gradient (slope) and intercept
            gradients[i, j] = model.coef_[0]
            intercepts[i, j] = model.intercept_

    return gradients, intercepts

# Usage example
# gradients, intercepts = calculate_avo_attributes(data, offsets)


def plot_avo_attributes(gradients, intercepts, slice_index=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    # If a specific inline or crossline slice is given, plot that slice; otherwise, plot the entire volume's mean
    if slice_index is not None:
        grad_plot = gradients[:, slice_index, :]
        inter_plot = intercepts[:, slice_index, :]
    else:
        grad_plot = np.mean(gradients, axis=1)
        inter_plot = np.mean(intercepts, axis=1)
    
    im1 = axes[0].imshow(grad_plot, aspect='auto', cmap='viridis', extent=[offsets.min(), offsets.max(), 0, grad_plot.shape[0]])
    fig.colorbar(im1, ax=axes[0], orientation='vertical')
    axes[0].set_title('AVO Gradients')
    axes[0].set_xlabel('Offset')
    axes[0].set_ylabel('Sample Index')
    
    im2 = axes[1].imshow(inter_plot, aspect='auto', cmap='viridis', extent=[offsets.min(), offsets.max(), 0, inter_plot.shape[0]])
    fig.colorbar(im2, ax=axes[1], orientation='vertical')
    axes[1].set_title('AVO Intercepts')
    axes[1].set_xlabel('Offset')
    axes[1].set_ylabel('Sample Index')
    
    plt.tight_layout()
    plt.show()

# Example plotting call, assuming a middle slice for a 3D cube
# plot_avo_attributes(gradients, intercepts, slice_index=data.shape[1]//2)
def detect_anomalous_avo_classes(gradients, intercepts, gradient_thresholds, intercept_thresholds):
    """
    Detect anomalous AVO classes based on thresholds for gradients and intercepts.
    
    Args:
    gradients (numpy.ndarray): Array of AVO gradients.
    intercepts (numpy.ndarray): Array of AVO intercepts.
    gradient_thresholds (dict): Dictionary with gradient thresholds for each AVO class.
    intercept_thresholds (dict): Dictionary with intercept thresholds for each AVO class.
    
    Returns:
    numpy.ndarray: Array of AVO class labels (1, 2, 3, 4, 0 where 0 means no significant anomaly).
    """
    # Initialize array to hold the AVO class labels, default is 0 (no significant anomaly)
    avo_classes = np.zeros_like(gradients, dtype=int)
    
    # Class 1: Positive gradient, strong positive intercept
    class1 = (gradients > gradient_thresholds['class1']) & (intercepts > intercept_thresholds['class1'])
    avo_classes[class1] = 1
    
    # Class 2: Near-zero to slightly positive gradient, positive intercept
    class2 = ((gradients >= gradient_thresholds['class2_lower']) & (gradients <= gradient_thresholds['class2_upper'])) & (intercepts > intercept_thresholds['class2'])
    avo_classes[class2] = 2
    
    # Class 3: Negative gradient, positive intercept
    class3 = (gradients < -gradient_thresholds['class3']) & (intercepts > intercept_thresholds['class3'])
    avo_classes[class3] = 3
    
    # Class 4: Positive gradient, negative or near-zero intercept
    class4 = (gradients > gradient_thresholds['class4']) & (intercepts <= intercept_thresholds['class4'])
    avo_classes[class4] = 4
    
    return avo_classes

# # Example thresholds
# gradient_thresholds = {
#     'class1': 0.2,  # High positive gradients
#     'class2_lower': 0,  # Lower bound for near-zero gradient
#     'class2_upper': 0.1,  # Upper bound for near-zero gradient
#     'class3': 0.2,  # High negative gradients
#     'class4': 0.2   # High positive gradients for Class 4
# }
# intercept_thresholds = {
#     'class1': 0.3,  # High intercepts for Class 1
#     'class2': 0.3,  # High intercepts for Class 2
#     'class3': 0.3,  # High intercepts for Class 3
#     'class4': 0     # Near-zero or negative intercepts for Class 4
# }

# # Using the function
# avo_classes = detect_anomalous_avo_classes(gradients, intercepts, gradient_thresholds, intercept_thresholds)

# # Visualization
# plt.imshow(avo_classes, aspect='auto', cmap='viridis')
# plt.colorbar(label='AVO Class')
# plt.title('Anomalous AVO Class Distribution')
# plt.xlabel('Trace')
# plt.ylabel('Depth/Sample Index')
# plt.show()

