import tensorflow as tf
import numpy as np

class SOMTensorFlow:
    def __init__(self, height, width, input_dim, learning_rate=0.1):
        self.height = height
        self.width = width
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Initialize weights using TensorFlow variables
        self.weights = tf.Variable(tf.random.normal([height * width, input_dim]))
        self.locations = self._generate_location_matrix(height, width)

    def _generate_location_matrix(self, height, width):
        """
        Generate a matrix with the coordinates of each neuron in the grid.
        """
        return tf.constant(np.array(list(self._iterate_indices(height, width))), dtype=tf.float32)

    def _iterate_indices(self, height, width):
        """
        An iterator over the grid coordinates.
        """
        for i in range(height):
            for j in range(width):
                yield np.array([i, j])

    def _find_bmu(self, sample):
        """
        Find the best matching unit (BMU) for a given sample.
        """
        expanded_sample = tf.expand_dims(sample, 0)
        squared_diff = tf.square(self.weights - expanded_sample)
        distances = tf.reduce_sum(squared_diff, axis=1)
        bmu_index = tf.argmin(distances)
        bmu_location = tf.gather(self.locations, bmu_index)
        return bmu_location

    def _update_weights(self, sample, bmu_location, iteration, total_iterations):
        """
        Update the weights of the SOM.
        """
        # Learning rate and radius decay
        lr = self.learning_rate * tf.exp(-tf.cast(iteration, tf.float32) / tf.cast(total_iterations, tf.float32))
        radius = tf.cast(tf.maximum(self.height, self.width) / 2, tf.float32) * tf.exp(-tf.cast(iteration, tf.float32) / tf.cast(total_iterations, tf.float32))
        squared_radius = tf.square(radius)

        # Update weights
        location_diff = self.locations - bmu_location
        squared_distance = tf.reduce_sum(tf.square(location_diff), axis=1)
        neighbourhood_func = tf.exp(-squared_distance / (2 * squared_radius))
        learning_rate_factor = neighbourhood_func * lr

        # Reshape for broadcasting
        shaped_lr_factor = tf.reshape(learning_rate_factor, [self.height * self.width, 1])
        weight_difference = sample - self.weights
        weight_update = shaped_lr_factor * weight_difference

        # Apply update
        self.weights.assign_add(weight_update)

    def train(self, data, num_iterations):
        """
        Train the SOM on the given data.
        """
        for iteration in range(num_iterations):
            for sample in data:
                bmu_location = self._find_bmu(sample)
                self._update_weights(sample, bmu_location, iteration, num_iterations)
