import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import keras as ks
from keras import layers, models


class DeepLearningModels:
    def __init__(self, model_type, device='auto', **kwargs):
        self.device = device
        with self._select_device():
            self.model = self._initialize_model(model_type, **kwargs)
    
    def _select_device(self):
        """Specifying device:

        'auto' will automatically select a GPU if available; otherwise, it falls back to the CPU.
        'cpu' forces the model to run on the CPU.
        'gpu', 'gpu0', 'gpu1', etc., allows selection of specific GPUs if you have multiple GPUs available.

        """
        if self.device == 'auto':
            return tf.device('/device:GPU:0' if tf.test.is_gpu_available() else '/cpu:0')
        elif self.device.startswith('gpu'):
            return tf.device(f'/device:GPU:{self.device[-1]}')
        elif self.device == 'cpu':
            return tf.device('/cpu:0')
        else:
            raise ValueError(f"Unsupported device specified: {self.device}")
    
    def _initialize_model(self, model_type, **kwargs):
        if model_type == 'unet':
            return self._create_unet(**kwargs)
        elif model_type == 'lstm':
            return self._create_lstm(**kwargs)
        elif model_type == 'gru':
            return self._create_gru(**kwargs)
        elif model_type == 'pca':
            # Note: PCA is not a TensorFlow model; handled separately.
            return PCA(**kwargs)
        elif model_type == 'transformer':
            return self.model == self._create_transformer(input_shape, **kwargs)
        elif model_type == 'som':
            # SOM needs a custom or external implementation for TensorFlow.
            return None # Placeholder for SOM implementation.
        elif model_type == 'contrastive_1d_cnn':
            return self._create_contrastive_1d_cnn(**kwargs)
        elif model_type == 'contrastive_2d_cnn':
            return self._create_contrastive_2d_cnn(**kwargs)
        else:
            raise ValueError("Unsupported model type.")
    
    def _create_unet(self, input_shape, num_classes):
        # This is a placeholder. You should define your U-Net architecture here.
        inputs = layers.Input(shape=input_shape)

        # Encoder
        x1, p1 = self._encoder_block(inputs, 64)
        x2, p2 = self._encoder_block(p1, 128)
        x3, p3 = self._encoder_block(p2, 256)
        x4, p4 = self._encoder_block(p3, 512)

        # Bottleneck
        b = self._conv_block(p4, 1024)

        # Decoder
        d1 = self._decoder_block(b, x4, 512)
        d2 = self._decoder_block(d1, x3, 256)
        d3 = self._decoder_block(d2, x2, 128)
        d4 = self._decoder_block(d3, x1, 64)

        # Output layer
        outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(d4)

        model = models.Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _conv_block(self, input_tensor, num_filters):
        x = layers.Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def _encoder_block(self, input_tensor, num_filters):
        x = self._conv_block(input_tensor, num_filters)
        p = layers.MaxPooling2D((2, 2))(x)
        return x, p

    def _decoder_block(self, input_tensor, concat_tensor, num_filters):
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
        x = layers.concatenate([x, concat_tensor], axis=-1)
        x = self._conv_block(x, num_filters)
        return x


    

    def _create_lstm(self, input_shape, lstm_layers, output_units):
        """
        Function to build a deeper and customizable LSTM model.
        
        Args:
        - input_shape: Shape of the input data (time steps, features).
        - lstm_layers: A list of dictionaries, each specifying the 'units' and 'dropout' for an LSTM layer.
        - output_units: Number of units in the output layer.
        
        Returns:
        - A tf.keras Model instance.
        """
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        
        # Add LSTM layers based on the lstm_layers argument
        for i, layer_config in enumerate(lstm_layers):
            return_sequences = i < len(lstm_layers) - 1  # Only the last layer returns sequences=False
            model.add(layers.LSTM(layer_config['units'], return_sequences=return_sequences))
            if 'dropout' in layer_config and layer_config['dropout'] > 0:
                model.add(layers.Dropout(layer_config['dropout']))
        
        # Output layer
        model.add(layers.Dense(output_units, activation='softmax'))  # Change activation based on your application

        return model

    
    def _create_gru(self, input_shape, gru_layers, output_units, **kwargs):
            """
            Function to build a deeper and customizable GRU model.
            
            Args:
            - input_shape: Shape of the input data (time steps, features).
            - gru_layers: A list of dictionaries, each specifying the 'units' and 'dropout' for a GRU layer.
            - output_units: Number of units in the output layer.
            
            Returns:
            - A tf.keras Model instance.
            """
            model = models.Sequential()
            model.add(layers.Input(shape=input_shape))
            
            # Add GRU layers based on the gru_layers argument
            for i, layer_config in enumerate(gru_layers):
                return_sequences = i < len(gru_layers) - 1  # Only the last layer returns sequences=False
                model.add(layers.GRU(layer_config['units'], return_sequences=return_sequences))
                if 'dropout' in layer_config and layer_config['dropout'] > 0:
                    model.add(layers.Dropout(layer_config['dropout']))
            
            # Output layer
            model.add(layers.Dense(output_units, activation='softmax'))  # Adjust activation as needed
            
            return model
    
    def create_1d_cnn_base_model(input_shape):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu')  # Feature vector
        ])
        return model
    
    def create_2d_cnn_base_model(input_shape):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu')  # Feature vector
        ])
        return model

    
    def _create_contrastive_1d_cnn(self, input_shape, num_classes):
        # Placeholder for a 1D CNN architecture suitable for contrastive learning

        input_1 = layers.Input(input_shape)
        input_2 = layers.Input(input_shape)
        
        # Re-use the same instance of the model for both inputs
        base_model = self.create_1d_cnn_base_model(input_shape)
        encoded_1 = base_model(input_1)
        encoded_2 = base_model(input_2)
        
        # Optionally, add a custom layer to compute the contrastive loss / distance metric
        # For simplicity, we'll concatenate the features and let downstream tasks (e.g., a classifier) handle it
        merged_vector = layers.concatenate([encoded_1, encoded_2], axis=-1)
        
        # You can add more layers here based on your specific task (e.g., a classification layer for supervised tasks)
        
        # Define the model
        model = models.Model(inputs=[input_1, input_2], outputs=merged_vector)
        return model

        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=input_shape),
        #     # Define the 1D CNN layers here
        #     tf.keras.layers.Dense(num_classes, activation='softmax')
        # ])
        # return model
    
    def contrastive_loss(y_true, y_pred, margin=1):
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def _create_contrastive_2d_cnn(self, input_shape, num_classes):
        # Placeholder for a 2D CNN architecture suitable for contrastive learning

        input_1 = layers.Input(input_shape)
        input_2 = layers.Input(input_shape)
        
        # Re-use the same instance of the 2D CNN model for both inputs
        base_model = self.create_2d_cnn_base_model(input_shape)
        encoded_1 = base_model(input_1)
        encoded_2 = base_model(input_2)
        
        # Concatenate features
        merged_vector = layers.concatenate([encoded_1, encoded_2], axis=-1)
        
        # Define the model
        model = models.Model(inputs=[input_1, input_2], outputs=merged_vector)
        return model

        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=input_shape),
        #     # Define the 2D CNN layers here
        #     tf.keras.layers.Dense(num_classes, activation='softmax')
        # ])
        # return model


# Example usage:
model_params = {
    'input_shape': (100, 128),  # Example shape for LSTM
    'units': 64,
    'num_layers': 2
}
# Specify the device: 'cpu', 'gpu', 'gpu0', 'gpu1', etc., or 'auto' for automatic selection.
device = 'gpu'  # Change to 'cpu' to force using the CPU, or 'gpu' for GPU.
model = DeepLearningModels('lstm', device=device, **model_params)
model.model.summary()
