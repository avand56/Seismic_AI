from tensorflow import keras
from keras import layers
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband

class UNetHyperModel2D(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        inputs = keras.Input(shape=self.input_shape)

        ### Encoder
        c1 = layers.Conv2D(hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
                           (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
                           (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        ### Continuing the pattern for other layers
        # Add more layers similar to above with hyperparameters for each layer's filters

        ### Output Layer
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(c1)  # Assuming a simple end for example

        model = keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=keras.optimizers.Adam(
                      hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

def tune_unet2d(input_shape, num_classes, data):
    tuner = Hyperband(
        UNetHyperModel2D(input_shape, num_classes),
        objective='val_accuracy',
        max_epochs=10,
        directory='unet_tuning',
        project_name='unet_tuning'
    )

    tuner.search(data['train_images'], data['train_labels'], epochs=10, validation_split=0.2)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The best number of filters for the first conv layer is {best_hps.get('conv1_filters')}
    and the best learning rate for the optimizer is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the best hyperparameters and train it on the data for 50 epochs
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(data['train_images'], data['train_labels'], epochs=50, validation_split=0.2)
    return best_model, history


class UNet3DHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        inputs = layers.Input(shape=self.input_shape)

        # Encoder (Contracting Path)
        x = self.down_block(inputs, hp.Int('filters_1', 32, 128, step=32))
        x1 = self.down_block(x, hp.Int('filters_2', 64, 256, step=32))
        x2 = self.down_block(x1, hp.Int('filters_3', 128, 512, step=32))
        
        # Bottleneck
        b = layers.Conv3D(hp.Int('filters_bottleneck', 256, 1024, step=256), (3, 3, 3), activation='relu', padding='same')(x2)
        b = layers.Conv3D(hp.Int('filters_bottleneck', 256, 1024, step=256), (3, 3, 3), activation='relu', padding='same')(b)
        
        # Decoder (Expanding Path)
        x = self.up_block(b, x2, hp.Int('filters_4', 128, 512, step=32))
        x = self.up_block(x, x1, hp.Int('filters_5', 64, 256, step=32))
        x = self.up_block(x, x, hp.Int('filters_6', 32, 128, step=32))

        # Output layer
        outputs = layers.Conv3D(self.num_classes, (1, 1, 1), activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return model

    def down_block(self, x, filters):
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        return x

    def up_block(self, x, skip, filters):
        x = layers.Conv3DTranspose(filters, (3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        x = layers.concatenate([x, skip])
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        return x
    
def tune_unet3d(input_shape, num_classes, data):
    tuner = Hyperband(
        UNet3DHyperModel(input_shape, num_classes),
        objective='val_accuracy',
        max_epochs=10,
        directory='unet3d_tuning',
        project_name='unet3d_tuning'
    )

    tuner.search(data['train'], epochs=10, validation_data=data['val'])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return tuner.hypermodel.build(best_hps)
