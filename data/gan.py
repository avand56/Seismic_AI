import tensorflow as tf
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

class SeismicGAN:
    def __init__(self, img_shape):
        """
        Initializes the GAN model for seismic reflection data generation.

        Args:
        - img_shape (tuple): The shape of the seismic reflection images, e.g., (height, width, channels).
        """
        self.img_shape = img_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        """
        Builds the Generator model.
        """
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_dim=100),  # 100-dimensional noise vector as input
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(np.prod(self.img_shape), activation='tanh'),  # Output the image
            layers.Reshape(self.img_shape)
        ])
        return model

    def build_discriminator(self):
        """
        Builds the Discriminator model.
        """
        model = models.Sequential([
            layers.Flatten(input_shape=self.img_shape),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification: real or fake
        ])
        # Compile the discriminator
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gan(self):
        """
        Constructs the GAN by stacking the generator and discriminator.
        """
        # Make the discriminator not trainable when we are training the generator
        self.discriminator.trainable = False

        model = models.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, real_data, epochs, batch_size=128, noise_dim=100):
        """
        Train the GAN model.

        Args:
        - real_data (np.array): Real seismic reflection data images for training the Discriminator.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Size of the batches of data.
        - noise_dim (int): Dimensionality of the noise vector for the Generator.
        """
        # This method is a simplified outline. Adapt it as needed for your specific scenario.
        half_batch = batch_size // 2

        for epoch in range(epochs):
            # Select a random half-batch of real images
            idx = np.random.randint(0, real_data.shape[0], half_batch)
            imgs_real = real_data[idx]

            # Generate a half batch of fake images
            noise = np.random.normal(0, 1, (half_batch, noise_dim))
            imgs_fake = self.generator.predict(noise)

            # Prepare labels for real and fake images
            valid_y = np.ones((half_batch, 1))  # Labels for real images
            fake_y = np.zeros((half_batch, 1))  # Labels for fake images

            # Ensure TensorFlow is using the GPU, if available
            with tf.device('/GPU:0'):  # This specifies that TensorFlow should run the following operations on the GPU, if one is available.
                # Train the discriminator (real classified as ones and fakes as zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs_real, valid_y)
                d_loss_fake = self.discriminator.train_on_batch(imgs_fake, fake_y)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Generate a batch of noise for generator training
                noise = np.random.normal(0, 1, (batch_size, noise_dim))
                
                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))

            # Optionally, log the progress, e.g., print out losses or update a training history record
            print(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")

            # Optionally, save generated images at certain intervals for inspection
            if (epoch + 1) % 100 == 0:
                self.sample_images(epoch + 1)

    def sample_images(self, epoch, noise_dim=100, samples=5):
        """
        Saves generated seismic reflection data images to inspect the progress of the generator.

        Args:
        - epoch (int): The current epoch, used for naming the output file.
        - noise_dim (int): Dimensionality of the noise vector.
        - samples (int): Number of images to generate.
        """
        noise = np.random.normal(0, 1, (samples, noise_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(1, samples, figsize=(samples * 3, 3))
        for i in range(samples):
            axs[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
            axs[i].axis('off')
        fig.savefig(f"seismic_{epoch}.png")
        plt.close()

