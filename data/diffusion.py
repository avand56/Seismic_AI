import tensorflow as tf
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

class SeismicDiffusionModel(tf.keras.Model):
    def __init__(self):
        super(SeismicDiffusionModel, self).__init__()
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        # Define your encoder architecture
        return models.Sequential([
            layers.InputLayer(input_shape=(256, 256, 1)),  # Adjust based on your data
            layers.Conv2D(32, kernel_size=3, strides=2, activation='relu'),
            layers.Conv2D(64, kernel_size=3, strides=2, activation='relu'),
            layers.Flatten(),
            layers.Dense(256, activation='relu')
        ])

    def _build_decoder(self):
        # Define your decoder architecture
        return models.Sequential([
            layers.Dense(64 * 64, activation='relu'),
            layers.Reshape((64, 64, 64)),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


    def forward_process(self, images, num_steps=10):
        """
        Simulates the forward process on 2D images by gradually adding Gaussian noise.
        
        Args:
        - images: A TensorFlow tensor of shape (batch_size, height, width, channels).
        - num_steps: Number of steps to incrementally add noise.
        
        Returns:
        - A list of tensors representing the images at each step of the forward process.
        """
        noised_images = [images]
        for step in range(1, num_steps + 1):
            noise_level = np.sqrt(step / num_steps)
            noise = tf.random.normal(shape=images.shape, mean=0.0, stddev=noise_level)
            noised_image = images + noise
            noised_images.append(noised_image)
        return noised_images

    def inverse_process(self, noised_images, num_steps=10):
        """
        A naive attempt to reverse the forward process by subtracting the added noise.
        
        Args:
        - noised_images: A list of tensors from the forward process.
        - num_steps: Number of steps to attempt to remove noise.
        
        Returns:
        - A list of tensors representing the attempt to reconstruct the images at each step.
        """
        reconstructed_images = [noised_images[-1]]
        for step in reversed(range(1, num_steps + 1)):
            noise_level = np.sqrt(step / num_steps)
            noise = tf.random.normal(shape=noised_images[0].shape, mean=0.0, stddev=noise_level)
            reconstructed_image = reconstructed_images[-1] - noise
            reconstructed_images.append(reconstructed_image)
        return reconstructed_images[::-1]


    def train_step(self,model, images, optimizer, loss_fn, num_noising_steps=10):
        """
        Performs a single training step, including the forward and inverse process.
        
        Args:
        - model: The diffusion model to be trained.
        - images: A batch of original seismic images.
        - optimizer: Optimizer to use for training.
        - loss_fn: Loss function for measuring the reconstruction quality.
        - num_noising_steps: Number of steps to apply in the noising process.
        """
        with tf.GradientTape() as tape:
            # Apply forward noising process
            noised_images = self.forward_process(images, num_steps=num_noising_steps)[-1]
            
            # Attempt to denoise using the model
            denoised_images = model(noised_images, training=True)
            
            # Calculate loss between the denoised images and the original images
            loss = loss_fn(images, denoised_images)
        
        # Calculate gradients and update model weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss

    # Example training loop
    def train_model(self,model, dataset, optimizer, loss_fn, epochs=1, num_noising_steps=10):
        for epoch in range(epochs):
            for step, images in enumerate(dataset):
                loss = self.train_step(model, images, optimizer, loss_fn, num_noising_steps=num_noising_steps)
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy()}")


# Initialize the model
model = SeismicDiffusionModel()

# Assume optimizer and loss function are defined
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.MeanSquaredError()


# Assuming `dataset` is a TensorFlow dataset of seismic images
model.train_model(model, dataset, optimizer, loss_fn, epochs=10, num_noising_steps=10)


# Example usage with a sample image
(sample_images, _), _ = tf.keras.datasets.mnist.load_data()
sample_images = sample_images[:10] / 255.0  # Normalize
sample_images = np.expand_dims(sample_images, axis=-1)  # Add channel dimension

# Convert to TensorFlow tensor
images_tensor = tf.convert_to_tensor(sample_images, dtype=tf.float32)

# Forward process
noised_images = model.forward_process(images_tensor, num_steps=10)

# Inverse process
reconstructed_images = model.inverse_process(noised_images, num_steps=10)

# Display original, noised, and reconstructed image
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(images_tensor[0,...,0], cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Noised")
plt.imshow(noised_images[-1][0,...,0], cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Reconstructed")
plt.imshow(reconstructed_images[-1][0,...,0], cmap='gray')
plt.show()
