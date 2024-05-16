import numpy as np
import tensorflow as tf
from keras import layers, models

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model=d_model)
        # apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, num_features, model_dim, num_heads, num_encoder_layers, num_classes, dropout_rate=0.1, max_length=100):
        super(TimeSeriesTransformer, self).__init__()
        self.num_features = num_features
        self.model_dim = model_dim

        # Embedding layer for time series features
        self.embedding = layers.Dense(model_dim)
        self.dropout = layers.Dropout(dropout_rate)

        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(max_length, model_dim)

        # Transformer Encoder layers
        self.transformer_encoder = layers.TransformerEncoder(
            num_layers=num_encoder_layers,
            embed_dim=model_dim,
            num_heads=num_heads,
            ff_dim=model_dim * 4,  # Feedforward network dimension
            dropout_rate=dropout_rate
        )
        
        # Global average pooling to reduce sequence dimension
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Final classification layer with dropout
        self.final_dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x = self.positional_encoding(x)
        
        x = self.transformer_encoder(x)
        
        x = self.global_pool(x)
        x = self.final_dropout(x)
        return self.classifier(x)
    
    def train_model(model, train_dataset, val_dataset, epochs=10):
        """
        Train the TimeSeriesTransformer model.

        Args:
        - model: The TimeSeriesTransformer model instance to be trained.
        - train_dataset (tf.data.Dataset): The dataset for training, including features and labels.
        - val_dataset (tf.data.Dataset): The dataset for validation, including features and labels.
        - epochs (int): The number of epochs to train the model.
        """
        # Fit the model to the data
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs
        )
        
        return history

#########
# to run
#########    

# # Example usage with dropout and positional encodings
# num_features = 10  # Number of features in your time series data
# model_dim = 64  # Dimensionality of model embeddings
# num_heads = 4  # Number of attention heads
# num_encoder_layers = 6  # Number of encoder layers
# num_classes = 3  # Number of classes for classification
# dropout_rate = 0.1  # Dropout rate
# max_length = 100  # Maximum length of the time series sequences


# model = TimeSeriesTransformer(num_features, model_dim, num_heads, num_encoder_layers, num_classes, dropout_rate, max_length)
# model.build(input_shape=(None, max_length, num_features))  # input shape
# model.summary()

# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# loss = tf.keras.losses.CategoricalCrossentropy()
# metrics = ['accuracy']

# model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
