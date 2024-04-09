import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import keras as ks
from keras import layers, models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import tensorflow_probability as tfp
import umap 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class ContrastiveTimeSeriesModel:
    def __init__(self, input_shape, feature_dim=64):
        """
        Initializes the contrastive learning model for time series data.

        Args:
        - input_shape (tuple): Shape of the input time series data (time_steps, num_features).
        - feature_dim (int): Dimensionality of the output embedding.
        """
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the shared 1D CNN architecture for feature extraction.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        x = TimeSeriesAugmentation()(inputs, training=True)
        x = layers.Conv1D(32, 7, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(self.feature_dim, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    

    def _build_model_3d(self, input_shape, feature_dim):
        """
        Builds the 3D CNN architecture for feature extraction.

        Args:
        - input_shape (tuple): Shape of the input data.
        - feature_dim (int): Dimensionality of the output embedding.

        Returns:
        - A Keras model representing the base 3D CNN architecture.
        """
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
            layers.MaxPooling3D(pool_size=(2, 2, 2)),
            layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
            layers.MaxPooling3D(pool_size=(2, 2, 2)),
            layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'),
            layers.GlobalAveragePooling3D(),
            layers.Dense(feature_dim, activation=None),  # No activation on final embedding layer
        ])
        return model

    def compile(self, optimizer, loss):
        """
        Configures the model for training.

        Args:
        - optimizer: String (name of optimizer) or optimizer instance.
        - loss: A loss instance or a callable.
        """
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, dataset, epochs=1, batch_size=32):
        """
        Trains the model on the provided dataset.

        Args:
        - dataset: A tf.data.Dataset object that yields pairs of time series data and a label indicating similarity.
        - epochs (int): Number of epochs to train for.
        - batch_size (int): Batch size for training.
        """
        # Assuming the dataset yields ([time_series_1, time_series_2], label) pairs
        self.model.fit(dataset.batch(batch_size), epochs=epochs)

    def evaluate(self, dataset, batch_size=32):
        """
        Evaluates the model on the provided dataset.

        Args:
        - dataset: A tf.data.Dataset object for evaluation.
        - batch_size (int): Batch size for evaluation.
        """
        return self.model.evaluate(dataset.batch(batch_size))
    

    # Define a simple contrastive loss function for demonstration
    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def prepare_dataset_with_metadata(data, spatial_metadata):
        """
        Prepare the dataset and associate each data point with spatial metadata.

        Args:
        - data: A list or array of data points (e.g., time series, images).
        - spatial_metadata: A list of spatial metadata corresponding to each data point. 
                            This could be (latitude, longitude) tuples or any form of spatial identifiers.

        Returns:
        - A list of tuples, where each tuple contains a data point and its corresponding spatial metadata.
        """
        dataset_with_metadata = list(zip(data, spatial_metadata))
        return dataset_with_metadata
    
    def map_anomalies_to_locations(anomalies_indices, dataset_with_metadata):
        """
        Map detected anomalies to their corresponding spatial locations using the recorded metadata.

        Args:
        - anomalies_indices: Indices of data points identified as anomalies.
        - dataset_with_metadata: The dataset including spatial metadata, as prepared by `prepare_dataset_with_metadata`.

        Returns:
        - A list of spatial metadata for the anomalies.
        """
        anomalies_locations = [dataset_with_metadata[i][1] for i in anomalies_indices]
        return anomalies_locations

    def map_clusters_to_locations(cluster_assignments, dataset_with_metadata):
        """
        Map each cluster to its corresponding spatial locations using the recorded metadata.

        Args:
        - cluster_assignments: Cluster assignments for each data point.
        - dataset_with_metadata: The dataset including spatial metadata, as prepared by `prepare_dataset_with_metadata`.

        Returns:
        - A dictionary mapping each cluster to a list of its spatial metadata.
        """
        clusters_locations = {}
        for idx, cluster_id in enumerate(cluster_assignments):
            if cluster_id not in clusters_locations:
                clusters_locations[cluster_id] = []
            clusters_locations[cluster_id].append(dataset_with_metadata[idx][1])
        return clusters_locations



    def enhanced_triplet_loss(batch_margin=0.5):
        """
        Triplet loss with hard negative mining and dynamic batch margin.

        Args:
        - batch_margin: Base margin for the loss, dynamically adjusted based on the batch.
        
        Returns:
        - A loss function compatible with Keras model compilation.
        """
        def loss(y_true, y_pred):
            anchor, positive, negative = tf.unstack(tf.reshape(y_pred, (-1, 3, y_pred.shape[-1])), num=3, axis=1)
            
            # Compute pairwise distance in the embedding space
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            
            # Find the hardest negatives within the batch
            # Negative is 'hard' if it is closer to the anchor than the positive is, but still further away than margin
            hardest_neg_dist = tf.reduce_min(neg_dist[neg_dist > pos_dist])
            
            # Dynamic margin adjustment: margin is increased if the hardest negative is closer than batch_margin to the positive
            dynamic_margin = tf.cond(hardest_neg_dist < batch_margin, lambda: batch_margin + (batch_margin - hardest_neg_dist), lambda: batch_margin)
            
            # Calculate triplet loss
            basic_loss = tf.maximum(pos_dist - neg_dist + dynamic_margin, 0.0)
            loss = tf.reduce_mean(basic_loss)
            
            return loss
        
        return loss
    

    def generate_contrastive_pairs(self, dataset, window_size, step_size, num_negative_pairs, augmentation_func=None):
        """
        Generate positive and negative pairs for contrastive learning from a dataset of windowed time series data.
        Note this does not work with triplet loss.

        Args:
        - dataset: A tf.data.Dataset instance containing time series data.
        - window_size (int): The size of each window/subsequence to generate.
        - step_size (int): The step size between windows for the sliding window mechanism.
        - num_negative_pairs (int): The number of negative pairs to generate for each positive pair.
        - augmentation_func (callable): A function to apply augmentation to a window to generate a positive pair.

        Returns:
        - A tf.data.Dataset instance containing pairs (positive and negative) and labels (1 for positive, 0 for negative).
        """
        def create_windows(time_series):
            return time_series.window(size=window_size, shift=step_size, drop_remainder=True).flat_map(lambda w: w.batch(window_size))
        
        def create_pairs(subsequence):
            # Positive pair generation through augmentation
            pos_pair = tf.stack([subsequence, augmentation_func(subsequence)], axis=0)  # Shape: (2, window_size, num_features)
            pos_label = tf.constant([1], dtype=tf.int32)
            
            # Negative pair placeholders (to be generated later)
            neg_pairs = tf.TensorArray(dtype=tf.float32, size=num_negative_pairs, dynamic_size=False)
            neg_labels = tf.TensorArray(dtype=tf.int32, size=num_negative_pairs, dynamic_size=False)
            
            # Dummy operation, replace with actual negative pair generation logic
            for i in tf.range(num_negative_pairs):
                neg_pairs = neg_pairs.write(i, tf.stack([subsequence, tf.zeros_like(subsequence)], axis=0))  # Placeholder
                neg_labels = neg_labels.write(i, tf.constant([0], dtype=tf.int32))
            
            return tf.data.Dataset.from_tensor_slices(((tf.concat([pos_pair[tf.newaxis], neg_pairs.stack()], axis=0), 
                                                        tf.concat([pos_label, neg_labels.stack()], axis=0))))

        # Apply windowing and map each window to create pairs
        windowed_dataset = dataset.flat_map(create_windows)
        pair_dataset = windowed_dataset.flat_map(create_pairs)

        return pair_dataset


    def generate_triplets(dataset, window_size=None, step_size=None, num_triplets=1000, augmentation_func=None):
        """
        Generate triplets for contrastive learning from a dataset of time series data,
        with optional windowing.
        
        Args:
        - dataset: A tf.data.Dataset instance containing time series data.
        - window_size: The size of each window/subsequence to generate (optional).
        - step_size: The step size between windows for the sliding window mechanism (optional).
        - num_triplets: Number of triplets to generate.
        - augmentation_func: Function to apply augmentation to generate a positive example.
        
        Returns:
        - A tf.data.Dataset instance containing triplets (anchor, positive, negative).
        """
        def create_windows(time_series):
            # Assume the function returns a dataset of windows if windowing is applied
            return time_series.window(size=window_size, shift=step_size, drop_remainder=True).flat_map(lambda w: w.batch(window_size))
        
        def create_triplets(subsequence):
            # Anchor and positive are the same subsequence with and without augmentation
            anchor = subsequence
            positive = augmentation_func(subsequence) if augmentation_func else subsequence
            
            # Assuming dataset provides multiple examples to choose a negative from
            # The negative example should be different from the anchor/positive
            negative = dataset.filter(lambda x: not tf.reduce_all(tf.equal(x, subsequence))).take(1)
            
            # Return a triplet (anchor, positive, negative)
            return tf.data.Dataset.from_tensors((anchor, positive, negative))
        
        # Apply optional windowing
        if window_size and step_size:
            windowed_dataset = dataset.flat_map(lambda x: create_windows(x))
        else:
            windowed_dataset = dataset
        
        # Generate triplets
        triplet_dataset = windowed_dataset.flat_map(create_triplets).take(num_triplets)
        
        return triplet_dataset


    # simple augmentation function: adding random noise
    def simple_augmentation(self, subsequence):
        noise = tf.random.normal(shape=tf.shape(subsequence), mean=0.0, stddev=0.05)
        return subsequence + noise

    def get_embeddings(model, dataset):
        embeddings = model.predict(dataset)
        return embeddings
    
    def cluster_embeddings(self,embeddings, method='kmeans', n_clusters=5, linkage_method='ward'):
        """
        Cluster embeddings using K-means or hierarchical clustering.

        Args:
        - embeddings (np.array): The embeddings to cluster, shape (n_samples, n_features).
        - method (str): The clustering method to use ('kmeans' or 'hierarchical').
        - n_clusters (int): The number of clusters to form.
        - linkage_method (str): The linkage criterion for hierarchical clustering ('ward', 'single', 'complete', 'average').

        Returns:
        - cluster_labels (np.array): An array of cluster labels for each embedding.
        """
        if method == 'kmeans':
            # Use K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
        elif method == 'hierarchical':
            # Use hierarchical/agglomerative clustering
            Z = linkage(embeddings, method=linkage_method)
            cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        else:
            raise ValueError("Unsupported clustering method. Choose 'kmeans' or 'hierarchical'.")

        return cluster_labels


class TimeSeriesAugmentation(tf.keras.layers.Layer):
    def __init__(self, noise_level=0.01, magnitude_warp=0.02, time_warp=0.02, **kwargs):
        super(TimeSeriesAugmentation, self).__init__(**kwargs)
        self.noise_level = noise_level
        self.magnitude_warp = magnitude_warp
        self.time_warp = time_warp

    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # Apply random noise
        noisy_data = inputs + tf.random.normal(tf.shape(inputs), stddev=self.noise_level)

        # Apply random time warping
        time_warped_data = self.apply_time_warp(noisy_data, self.time_warp)
        
        # Apply random magnitude warping
        output = self.apply_magnitude_warp(time_warped_data, self.magnitude_warp)
        
        return output

    def apply_time_warp(self, data, time_warping_factor):
        seq_len = tf.shape(data)[1]
        num_features = tf.shape(data)[-1]
        warp_curve = self.generate_time_warp_curve(seq_len, time_warping_factor, num_features)
        
        # Apply the time warp curve using interpolation
        original_indices = tf.range(seq_len)
        # Use TensorFlow's interpolation functions, adjusting indices based on the warp curve
        warped_data = tf.gather(data, warp_curve, axis=1, batch_dims=1)
        return warped_data
    
    def generate_time_warp_curve(seq_len, warp_factor=0.1, num_knots=4):
        """
        Generates a time warp curve using linear interpolation between knots.
        
        Args:
        - seq_len: The length of the time series.
        - warp_factor: The factor by which to warp the time series.
        - num_knots: The number of knots to use for the piecewise linear function.
        
        Returns:
        - A 1D TensorFlow tensor representing the warped indices for the time series.
        """
        # Ensure warp_factor is positive and less than 1 for meaningful warping
        warp_factor = tf.clip_by_value(warp_factor, 0, 1)
        
        # Original indices
        original_indices = tf.linspace(0.0, tf.cast(seq_len - 1, tf.float32), seq_len)
        
        # Knot positions along the original sequence
        knot_indices = tf.linspace(0.0, tf.cast(seq_len - 1, tf.float32), num_knots)
        
        # Warped positions for each knot
        # The first and last knots are not warped to keep the sequence length unchanged
        warp_offsets = tf.random.uniform((num_knots,), -warp_factor * seq_len, warp_factor * seq_len)
        warp_offsets = tf.tensor_scatter_nd_update(warp_offsets, [[0], [num_knots-1]], [[0.0], [0.0]])
        warped_knot_indices = knot_indices + warp_offsets
        
        # Interpolate a warp curve between knots
        warped_indices = tfp.math.interp_regular_1d_grid(original_indices, x_ref_min=0.0, x_ref_max=tf.cast(seq_len - 1, tf.float32), y_ref=warped_knot_indices, fill_value='constant')
        
        # Ensure the indices are within bounds and rounded to integer values for indexing
        warped_indices = tf.clip_by_value(warped_indices, 0, seq_len - 1)
        warped_indices = tf.cast(tf.round(warped_indices), tf.int32)
        
        return warped_indices


    def generate_warp_curve(self, seq_len, magnitude_warping_factor):
        # This is a simplified version. You might want to generate a smooth curve,
        # for example, using a spline that slightly varies around 1.0.
        random_factors = tf.random.uniform((seq_len,), 1.0 - magnitude_warping_factor, 1.0 + magnitude_warping_factor)
        return random_factors

    def apply_magnitude_warp(self, data, magnitude_warping_factor):
        seq_len = tf.shape(data)[1]
        warp_curve = self.generate_warp_curve(seq_len, magnitude_warping_factor)
        return data * warp_curve

    def get_config(self):
        config = super(TimeSeriesAugmentation, self).get_config()
        config.update({
            "noise_level": self.noise_level,
            "magnitude_warp": self.magnitude_warp,
            "time_warp": self.time_warp,
        })
        return config
    
    def visualize_clusters(embeddings, cluster_labels, method='umap', random_state=42):
        """
        Visualize high-dimensional embeddings and their clusters using t-SNE or UMAP.

        Args:
        - embeddings (np.array): High-dimensional embeddings, shape (n_samples, n_features).
        - cluster_labels (np.array): Cluster labels for each embedding.
        - method (str): Dimensionality reduction method to use ('tsne' or 'umap').
        - random_state (int): Random state for reproducibility.
        """
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=random_state)
        elif method == 'umap':
            reducer = umap.UMAP(random_state=random_state)
        else:
            raise ValueError("Unsupported method. Choose 'tsne' or 'umap'.")

        embeddings_reduced = reducer.fit_transform(embeddings)

        # Plotting
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], c=cluster_labels, cmap='Spectral', s=5)
        plt.colorbar(scatter)
        plt.title(f'{method.upper()} Visualization of Clusters')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.show()

# Example: Compute similarity matrix
# embeddings = get_embeddings(model, data_for_prediction)
# similarity_matrix = cosine_similarity(embeddings)



###############
# run model
###############
                
# Assuming `parsed_dataset` is your dataset loaded and parsed from TFRecords
# window_size = 50
# step_size = 25
# num_negative_pairs = 2  # Example: For each positive pair, generate 2 negative pairs

# Generate the dataset of pairs
# pairs_dataset = model.generate_contrastive_pairs(parsed_dataset, window_size, step_size, num_negative_pairs, augmentation_func=model.simple_augmentation)


# Convert to tf.data.Dataset
#dataset = tf.data.Dataset.from_tensor_slices((pairs, labels))

# Assuming `train_dataset` is a tf.data.Dataset object that yields ([time_series_1, time_series_2], similarity_label) pairs
# model.train(train_dataset, epochs=10, batch_size=32)


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




#####################
# working with spatial
######################

# data = [...]  # Your dataset
# spatial_metadata = [(lat, lon) for lat, lon in zip(lats, lons)]  # List of (latitude, longitude)

# # Prepare the dataset with spatial metadata
# dataset_with_metadata = prepare_dataset_with_metadata(data, spatial_metadata)

# # After processing (e.g., anomaly detection), assuming you have anomalies_indices
# anomalies_indices = [...]  # Indices of detected anomalies
# anomalies_locations = map_anomalies_to_locations(anomalies_indices, dataset_with_metadata)

# # Or, if you performed clustering and have cluster_assignments
# cluster_assignments = [...]  # Cluster assignments for each data point
# clusters_locations = map_clusters_to_locations(cluster_assignments, dataset_with_metadata)
