import tensorflow as tf
import keras as ks
from keras import layers, models, callbacks
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import datetime
import matplotlib.pyplot as plt
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam


def create_segmentation_vgg16_model(input_shape, num_classes):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Upsampling and final layer to match the input size
        layers.Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(num_classes, (3, 3), activation='softmax', padding='same')  # Number of classes
    ])
    
    return model


def unet_model(input_shape, num_classes):
    inputs = layers.Input(input_shape)

    # Contracting Path (Encoder)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Expansive Path (Decoder)
    u6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output Layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


class UNet(models.Model):
    def __init__(self, input_shape, num_classes, filters, dimensionality=2):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        if dimensionality == 2:
            Conv = layers.Conv2D
            ConvTranspose = layers.Conv2DTranspose
            MaxPooling = layers.MaxPooling2D
        elif dimensionality == 3:
            Conv = layers.Conv3D
            ConvTranspose = layers.Conv3DTranspose
            MaxPooling = layers.MaxPooling3D
        else:
            raise ValueError("Dimensionality must be either 2 or 3.")

        # Setup the architecture
        self.build_unet(Conv, ConvTranspose, MaxPooling, input_shape, filters)

        # Set up TensorBoard logging
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1, write_graph=True)

    def build_unet(self, Conv, ConvTranspose, MaxPooling, input_shape, filters):
        inputs = layers.Input(shape=input_shape)
        x = inputs
        skips = []

        # Encoder
        for filter_count in filters:
            x = Conv(filter_count, 3, activation='relu', padding='same')(x)
            x = Conv(filter_count, 3, activation='relu', padding='same')(x)
            skips.append(x)
            x = MaxPooling(2)(x)

        # Bridge
        x = Conv(filters[-1] * 2, 3, activation='relu', padding='same')(x)
        x = Conv(filters[-1] * 2, 3, activation='relu', padding='same')(x)

        # Decoder
        for i in reversed(range(len(filters))):
            x = ConvTranspose(filters[i], 2, strides=2, padding='same')(x)
            x = layers.Concatenate()([x, skips.pop()])
            x = Conv(filters[i], 3, activation='relu', padding='same')(x)
            x = Conv(filters[i], 3, activation='relu', padding='same')(x)

        outputs = Conv(self.num_classes, 1, activation='softmax', padding='same')(x)
        self.model = models.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)

    def compile_and_fit(self, train_dataset, val_dataset, epochs=10, batch_size=32):
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.fit(train_dataset.batch(batch_size), epochs=epochs,
                 validation_data=val_dataset.batch(batch_size),
                 callbacks=[self.tensorboard_callback])
    

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




###########
# command line
##############
# def load_dataset(dataset_path, test_size=0.2):
#     # Placeholder function for loading and preprocessing your dataset
#     # Replace this with actual dataset loading code
#     images = np.random.rand(100, 256, 256, 3).astype(np.float32)  # Example synthetic data
#     labels = np.random.randint(0, 10, size=(100, 256, 256, 1)).astype(np.float32)  # Example synthetic labels
    
#     X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size)
#     train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(args.batch_size)
#     val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size)
#     return train_dataset, val_dataset

# def main(args):
#     # Initialize the U-Net model
#     model = ResNetSegmentation(input_shape=(256, 256, 3), num_classes=args.num_classes)
    
#     # Load the dataset
#     train_dataset, val_dataset = load_dataset(args.dataset_path, test_size=args.test_size)
    
#     # Train the model
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs)
    
#     # Optionally, visualize predictions on the validation dataset
#     if args.visualize:
#         model.visualize_predictions(val_dataset, num_examples=5)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train a ResNet-based segmentation model on 2D images.')
#     parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory.')
#     parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
#     parser.add_argument('--num_classes', type=int, required=True, help='Number of segmentation classes.')
#     parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used as validation set.')
#     parser.add_argument('--visualize', action='store_true', help='Whether to visualize model predictions.')

#     args = parser.parse_args()
    
#     main(args)

# python train_resnet_segmentation.py --dataset_path "/path/to/your/dataset" --num_classes 10 --epochs 20 --batch_size 16 --visualize
