from obspy.io.segy.segy import _read_segy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.contrastive_learning import ContrastiveTimeSeriesModel
import tensorflow as tf

def read_segy_file(file_path):
    """
    Reads a SEG-Y file and returns the seismic data. Note trace lengths need to be the same.
    
    Args:
    - file_path (str): The path to the SEG-Y file.
    
    Returns:
    - data (np.array): A 2D numpy array of seismic traces.
    """
    stream = _read_segy(file_path)
    data = np.stack([trace.data for trace in stream.traces])
    return data


def visualize_seismic_data(seismic_data):
    """
    Visualizes seismic data using a heatmap.
    
    Args:
    - seismic_data (np.array): A 2D numpy array of seismic traces.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(seismic_data, aspect='auto', cmap='seismic', extent=[0, seismic_data.shape[1], seismic_data.shape[0], 0])
    plt.colorbar(label='Amplitude')
    plt.xlabel('Trace')
    plt.ylabel('Time Sample')
    plt.title('Seismic Data')
    plt.show()

def find_max_trace_length(file_path):
    """
    Finds the maximum trace length in a SEG-Y file.
    
    Args:
    - file_path (str): Path to the SEG-Y file.
    
    Returns:
    - max_length (int): The maximum trace length found in the file.
    """
    stream = _read_segy(file_path)
    max_length = max(len(trace.data) for trace in stream.traces)
    return max_length


def preprocess_traces(file_path, max_length, padding_value=0):
    """
    Preprocesses traces in a SEG-Y file to have uniform length by padding shorter traces.
    
    Args:
    - file_path (str): Path to the SEG-Y file.
    - max_length (int): The uniform length to standardize all traces to.
    - padding_value (float): The value to use for padding shorter traces.
    
    Returns:
    - data (np.array): A 2D numpy array of preprocessed seismic traces.
    """
    stream = _read_segy(file_path)
    data = []
    for trace in stream.traces:
        trace_data = trace.data
        if len(trace_data) < max_length:
            # Pad shorter traces
            padding = np.full((max_length - len(trace_data),), padding_value)
            trace_data = np.concatenate([trace_data, padding])
        data.append(trace_data)
    data= np.array(data)
    return data

import numpy as np

def reshape_into_subsequences(data, sequence_length, step_size=None):
    """
    Reshape a time series dataset into subsequences.
    
    Args:
    - data (np.array): The original dataset of shape (num_timesteps, num_features).
    - sequence_length (int): The desired length of each subsequence.
    - step_size (int): The step size between subsequences for overlapping. If None, it defaults to sequence_length (non-overlapping).
    
    Returns:
    - subsequences (np.array): An array of subsequences of shape (n_subsequences, sequence_length, num_features).
    """
    if step_size is None:
        step_size = sequence_length  # Default to non-overlapping if step_size not provided

    num_timesteps, num_features = data.shape

    # Calculate the number of subsequences and initialize an array for them
    num_subsequences = (num_timesteps - sequence_length) // step_size + 1
    subsequences = []

    for start in range(0, num_timesteps - sequence_length + 1, step_size):
        end = start + sequence_length
        subsequences.append(data[start:end])

    return np.array(subsequences)

def scale_subsequences(subsequences):
    scaler = StandardScaler()
    # Reshape subsequences to (n_samples, n_features) for scaling
    subsequences_reshaped = subsequences.reshape(-1, subsequences.shape[-1])  # This line may need adjustment based on your exact data shape
    subsequences_scaled = scaler.fit_transform(subsequences_reshaped)
    # Reshape back to original shape if necessary
    subsequences_scaled = subsequences_scaled.reshape(subsequences.shape)
    return subsequences_scaled, scaler

def segment_into_sequences(data, sequence_length, step_size):
    """
    Segments the dataset into sequences of a fixed length using a sliding window.
    
    Args:
    - data (np.array): The dataset to segment, shape (num_timesteps, num_features).
    - sequence_length (int): The length of each sequence.
    - step_size (int): The step size to move the window across the dataset.
    
    Returns:
    - sequences (np.array): An array of shape (num_sequences, sequence_length, num_features).
    """
    sequences = []
    for start in range(0, data.shape[0] - sequence_length + 1, step_size):
        end = start + sequence_length
        sequences.append(data[start:end])
    return np.array(sequences)

def create_patches(data, patch_size, step):
    """
    Create patches from time series data by sliding a window across each time series.

    Args:
    - data (np.array): The full dataset, shape (num_samples, num_timesteps, num_features).
    - patch_size (int): The number of timesteps in each patch.
    - step (int): The step between the start of each patch.

    Returns:
    - patches (np.array): Patches created from the dataset, shape (num_patches, patch_size, num_features).
    """
    num_samples, num_timesteps, num_features = data.shape
    # Ensuring each series can be divided into patches with the given patch size and step
    if (num_timesteps - patch_size) % step != 0:
        # Adjust num_timesteps to fit the window exactly
        max_length = num_timesteps - (num_timesteps - patch_size) % step
        data = data[:, :max_length, :]

    # Calculate total number of patches
    num_patches_per_series = ((num_timesteps - patch_size) // step) + 1
    total_patches = num_patches_per_series * num_samples

    # Create patches
    patches = []
    for i in range(num_samples):
        for start in range(0, num_timesteps - patch_size + 1, step):
            end = start + patch_size
            patches.append(data[i, start:end, :])
    patches = np.array(patches)
    
    print(f"Total patches created: {total_patches}, actual patches array shape: {patches.shape}")
    return patches

def create_dataset(pairs, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((pairs, labels))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def generate_test_pairs(test_data, num_pairs):
    num_samples = test_data.shape[0]
    pairs = []
    labels = []

    # Generate positive pairs
    for _ in range(num_pairs // 2):
        idx = np.random.randint(0, num_samples)
        # For test pairs, typically no augmentation is applied
        pair = [test_data[idx], test_data[idx]]  # Using identical data or applying minimal transformation
        pairs.append(pair)
        labels.append(1)  # 1 represents a positive pair

    # Generate negative pairs
    for _ in range(num_pairs // 2):
        idx1, idx2 = np.random.choice(num_samples, size=2, replace=False)
        pair = [test_data[idx1], test_data[idx2]]
        pairs.append(pair)
        labels.append(0)  # 0 represents a negative pair

    return np.array(pairs), np.array(labels)

def train_contrastive_ts(dataset):
    # Instantiate and compile the model
    input_shape = (128, 1)  # Example input shape (time_steps, num_features=1)
    model = ContrastiveTimeSeriesModel(input_shape=input_shape, feature_dim=64)
    model.compile(optimizer='adam', loss=model.enhanced_triplet_loss(batch_margin=0.5))
    triplet_dataset= model.generate_triplets(dataset, window_size=None, step_size=None, num_triplets=1000, augmentation_func=None)
    model.fit(triplet_dataset, epochs=100, steps_per_epoch=200)
    embeddings = model.get_embeddings(model, dataset)
    # Cluster the embeddings using K-means
    cluster_labels_kmeans = model.cluster_embeddings(embeddings, method='kmeans', n_clusters=5)

    # Or cluster the embeddings using hierarchical clustering
    cluster_labels_hierarchical = model.cluster_embeddings(embeddings, method='hierarchical', n_clusters=5, linkage_method='ward')

    # Visualize using t-SNE
    model.visualize_clusters(embeddings, cluster_labels_kmeans, method='tsne')

    # Visualize using UMAP
    model.visualize_clusters(embeddings, cluster_labels_hierarchical, method='umap')