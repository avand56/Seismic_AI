import tensorflow as tf
from keras import layers, models
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

class UNet:
    def __init__(self, input_shape, num_classes, dimensionality=2):
        """
        Initializes a U-Net model for 2D or 3D data.

        Args:
        - input_shape (tuple): Shape of the input data, including channels. For 2D, it should be (height, width, channels), and for 3D, (depth, height, width, channels).
        - num_classes (int): The number of output classes.
        - dimensionality (int): The dimensionality of the data (2 for 2D data and 3 for 3D data).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dimensionality = dimensionality

        if dimensionality == 2:
            self.Conv = layers.Conv2D
            self.ConvTranspose = layers.Conv2DTranspose
            self.MaxPooling = layers.MaxPooling2D
            self.UpSampling = layers.UpSampling2D
        elif dimensionality == 3:
            self.Conv = layers.Conv3D
            self.ConvTranspose = layers.Conv3DTranspose
            self.MaxPooling = layers.MaxPooling3D
            self.UpSampling = layers.UpSampling3D
        else:
            raise ValueError("Dimensionality must be either 2 or 3.")

        self.model = self.build_model()

    def conv_block(self, inputs, num_filters):
        """Constructs a convolutional block."""
        x = self.Conv(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        x = self.Conv(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        return x

    def encoder_block(self, inputs, num_filters):
        """Constructs an encoder block."""
        x = self.conv_block(inputs, num_filters)
        p = self.MaxPooling((2, 2))(x) if self.dimensionality == 2 else self.MaxPooling((2, 2, 2))(x)
        return x, p

    def decoder_block(self, inputs, skip_features, num_filters):
        """Constructs a decoder block."""
        x = self.ConvTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs) if self.dimensionality == 2 else \
            self.ConvTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def build_model(self):
        """Builds the U-Net model."""
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        skips = []
        x = inputs
        for num_filters in [64, 128, 256, 512]:
            x, x_pool = self.encoder_block(x, num_filters)
            skips.append(x)
            x = x_pool

        # Bridge
        x = self.conv_block(x, 1024)

        # Decoder
        for i, num_filters in enumerate([512, 256, 128, 64]):
            x = self.decoder_block(x, skips[::-1][i], num_filters)

        # Output layer
        outputs = self.Conv(self.num_classes, 1, activation='softmax', padding='same')(x)

        model = models.Model(inputs, outputs, name="U-Net")
        return model
    def prepare_data(self, image_dir, mask_dir, image_size, num_classes, test_size=0.2, batch_size=32):
        """
        Prepares training and validation datasets from image and mask directories.

        Args:
        - image_dir (str): Directory containing the input images.
        - mask_dir (str): Directory containing the corresponding segmentation masks.
        - image_size (tuple): Desired size of the images and masks, e.g., (height, width).
        - num_classes (int): Number of classes in the segmentation task.
        - test_size (float): Proportion of the dataset to be used as validation data.
        - batch_size (int): Number of samples per batch.

        Returns:
        - train_dataset, val_dataset: Tuple of TensorFlow dataset objects for training and validation.
        """
        # Load and preprocess images and masks
        images, masks = [], []
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)  # Assuming corresponding mask has same filename

            image = load_img(image_path, target_size=image_size, color_mode='grayscale' if self.dimensionality == 2 else 'rgb')
            mask = load_img(mask_path, target_size=image_size, color_mode='grayscale')

            image = img_to_array(image) / 255.0  # Normalize images to [0, 1]
            mask = img_to_array(mask)[:,:,0] if self.dimensionality == 2 else img_to_array(mask)  # For 2D, ensure mask is a single channel
            mask = to_categorical(mask, num_classes=num_classes)  # One-hot encode masks

            images.append(image)
            masks.append(mask)

        images = np.array(images)
        masks = np.array(masks)

        # Split the dataset into training and validation
        X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=test_size, random_state=42)

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, val_dataset

class ResNetSegmentation(models.Model):
    def __init__(self, input_shape, num_classes):
        """
        Initializes a segmentation model using ResNet as the backbone.

        Args:
        - input_shape (tuple): The shape of the input images (height, width, channels).
        - num_classes (int): The number of classes for the segmentation task.
        """
        super(ResNetSegmentation, self).__init__()
        self.num_classes = num_classes
        
        # Load a pre-trained ResNet model as the encoder
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze the encoder layers
        base_model.trainable = False

        # Use the outputs of these layers as skip connections
        layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        self.encoder_layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the encoder model
        self.encoder = models.Model(inputs=base_model.input, outputs=self.encoder_layers, name='encoder')

        # Decoder part with upsampling
        self.decoder = self._build_decoder()

        # Output segmentation map
        self.segmentation_head = layers.Conv2D(num_classes, (1, 1), activation='softmax')

    def _build_decoder(self):
        """
        Builds the decoder part of the model.
        """
        decoder_filters = [256, 128, 64, 32]

        # Decoder model
        inputs = [layers.Input(shape=layer.shape[1:]) for layer in self.encoder_layers]
        x = inputs[-1]  # Start from the last encoder output
        for i, filters in enumerate(decoder_filters):
            if i > 0:  # Skip connections from encoder to decoder
                x = layers.Concatenate()([x, inputs[-i-1]])
            x = layers.Conv2DTranspose(filters, (3, 3), strides=2, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        return models.Model(inputs, x, name='decoder')

    def call(self, inputs):
        """
        Forward pass through the network.

        Args:
        - inputs: Input images.

        Returns:
        - Segmentation map for each input image.
        """
        encoder_outputs = self.encoder(inputs)
        x = self.decoder(encoder_outputs)
        return self.segmentation_head(x)

    def summary(self):
        print("Encoder:")
        self.encoder.summary()
        print("\nDecoder:")
        self.decoder.summary()
        print("\nSegmentation Head:")
        self.segmentation_head.summary()

    def train(self, train_dataset, val_dataset, epochs=10, batch_size=32):
        """
        Trains the model on the training dataset.

        Args:
        - train_dataset: A tf.data.Dataset object containing training data and labels.
        - val_dataset: A tf.data.Dataset object for validation.
        - epochs (int): The number of epochs to train for.
        - batch_size (int): The batch size for training.
        """
        self.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

        history = self.fit(train_dataset.batch(batch_size),
                           validation_data=val_dataset.batch(batch_size),
                           epochs=epochs)
        return history

    def evaluate(self, test_dataset, batch_size=32):
        """
        Evaluates the model on the test dataset.

        Args:
        - test_dataset: A tf.data.Dataset object containing test data and labels.
        - batch_size (int): The batch size for evaluation.
        """
        results = self.evaluate(test_dataset.batch(batch_size))
        print("Test loss, Test acc:", results)
        return results

    def visualize_predictions(self, dataset, num_examples=3):
        """
        Visualizes predictions for a subset of examples from the dataset.

        Args:
        - dataset: A tf.data.Dataset object containing data and labels.
        - num_examples (int): Number of examples to visualize.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, num_examples * 3))
        
        for images, labels in dataset.take(1):
            preds = self.predict(images)
            preds = tf.argmax(preds, axis=-1)
            for i in range(num_examples):
                plt.subplot(num_examples, 3, i * 3 + 1)
                plt.imshow(images[i].numpy().squeeze(), cmap='gray')
                plt.title("Input")
                plt.axis('off')

                plt.subplot(num_examples, 3, i * 3 + 2)
                plt.imshow(labels[i].numpy().squeeze(), cmap='gray')
                plt.title("True")
                plt.axis('off')

                plt.subplot(num_examples, 3, i * 3 + 3)
                plt.imshow(preds[i].numpy().squeeze(), cmap='gray')
                plt.title("Predicted")
                plt.axis('off')
        plt.tight_layout()
        plt.show()



###############
# compile model
###############
    
input_shape = (256, 256, 1)  # Example for grayscale images
num_classes = 3  # Example for 3 classes including background

unet_2d = UNet(input_shape=input_shape, num_classes=num_classes, dimensionality=2)
unet_2d.model.summary()  # To see the model architecture

unet_2d.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


############
# data prep
###########

image_dir = 'path/to/images/'
mask_dir = 'path/to/masks/'
image_size = (256, 256)  # Desired output size of images and masks
num_classes = 3  # Number of segmentation classes, including background

# Initialize the UNet model
model = UNet(input_shape=(*image_size, 1), num_classes=num_classes, dimensionality=2)  # Assuming grayscale images

# Prepare the data
train_dataset, val_dataset = model.prepare_data(image_dir, mask_dir, image_size, num_classes, test_size=0.2, batch_size=32)

# Now, `train_dataset` and `val_dataset` are ready to be used for training and validation.



##############
# Train resnet
##############
input_shape = (256, 256, 3)  # Adjust based on your data
num_classes = 10  # Adjust based on your task

model = ResNetSegmentation(input_shape=input_shape, num_classes=num_classes)

# Assuming train_dataset, val_dataset, and test_dataset are prepared tf.data.Dataset objects
history = model.train(train_dataset, val_dataset, epochs=20, batch_size=16)

model.evaluate(test_dataset, batch_size=16)

# Visualization on a part of the test dataset
model.visualize_predictions(test_dataset, num_examples=5)


###########
# command line
##############
def load_dataset(dataset_path, test_size=0.2):
    # Placeholder function for loading and preprocessing your dataset
    # Replace this with actual dataset loading code
    images = np.random.rand(100, 256, 256, 3).astype(np.float32)  # Example synthetic data
    labels = np.random.randint(0, 10, size=(100, 256, 256, 1)).astype(np.float32)  # Example synthetic labels
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size)
    return train_dataset, val_dataset

def main(args):
    # Initialize the U-Net model
    model = ResNetSegmentation(input_shape=(256, 256, 3), num_classes=args.num_classes)
    
    # Load the dataset
    train_dataset, val_dataset = load_dataset(args.dataset_path, test_size=args.test_size)
    
    # Train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs)
    
    # Optionally, visualize predictions on the validation dataset
    if args.visualize:
        model.visualize_predictions(val_dataset, num_examples=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet-based segmentation model on 2D images.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of segmentation classes.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used as validation set.')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize model predictions.')

    args = parser.parse_args()
    
    main(args)

# python train_resnet_segmentation.py --dataset_path "/path/to/your/dataset" --num_classes 10 --epochs 20 --batch_size 16 --visualize
