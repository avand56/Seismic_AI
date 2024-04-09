
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import MeanIoU
import numpy as np
from models.model_class import DeepLearningModels

# check if GPU recognized.
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))





def train_and_evaluate_model(model, train_dataset, val_dataset, epochs=100, batch_size=32, learning_rate=1e-4, model_save_path='best_model.h5', loss='categorical_crossentropy', metrics=['accuracy'], additional_callbacks=[]):
    # Compile the model with specified loss, metrics, and optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )

    # Default callbacks: EarlyStopping and ModelCheckpoint
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True)
    ] + additional_callbacks  # Append any additional callbacks provided as arguments

    history = model.fit(
        train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        validation_data=val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history):
    fig, axs = plt.subplots(2, 1, figsize=(8, 12))

    # Plot accuracy
    axs[0].plot(history.history['accuracy'], label='Training Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Training and Validation Accuracy')

    # Plot loss
    axs[1].plot(history.history['loss'], label='Training Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Training and Validation Loss')

    plt.show()


# # Assume `model` is an instance of TimeSeriesTransformer or another compatible model
# # Assume `train_dataset` and `val_dataset` are prepared tf.data.Dataset objects for training and validation

# # Train the model
# history = train_and_evaluate_model(model, train_dataset, val_dataset, epochs=100, batch_size=32, learning_rate=1e-4, model_save_path='best_time_series_transformer_model.h5')

# # Plot the training history
# plot_training_history(history)


##################################
# Train Contrastive Learning - 1D
#################################

# Assume `pairs` is your input data and `labels` indicates if pairs are similar (1) or not (0)
# model = create_contrastive_model(input_shape)
# model.compile(optimizer='adam', loss=contrastive_loss)
# model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=10)


##################################
# Train Contrastive Learning - 2D
#################################

# Assume `pairs` is your input data (pairs of images) and `labels` indicates if pairs are similar (1) or not (0)
# input_shape should match the dimensions of your images, e.g., (height, width, channels)
# model = create_contrastive_2d_model(input_shape)
# model.compile(optimizer='adam', loss=contrastive_loss)
# model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=10, batch_size=32)


######################
# U-Net
#############
# Example usage:
# input_shape = (256, 256, 3)  # Example input shape, change as needed
# num_classes = 3  # Example number of segmentation classes, change as needed
# model = DeepLearningModels('unet', input_shape, num_classes)
# model.model.summary()

def load_data():
    # Placeholder function to load your data
    # Replace with actual code to load your dataset
    # Return training and validation datasets
    return train_images, train_masks, val_images, val_masks

def preprocess_data(images, masks):
    # Placeholder for preprocessing steps
    # Normalize images, encode masks if necessary, etc.
    return images / 255.0, masks

def train_unet(model, train_images, train_masks, val_images, val_masks, epochs=10):
    # Preprocess the data
    train_images, train_masks = preprocess_data(train_images, train_masks)
    val_images, val_masks = preprocess_data(val_images, val_masks)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', MeanIoU(num_classes=3)])

    # Data augmentation
    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(train_images, augment=True, seed=seed)
    mask_datagen.fit(train_masks, augment=True, seed=seed)

    image_generator = image_datagen.flow(train_images, seed=seed)
    mask_generator = mask_datagen.flow(train_masks, seed=seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    # Train the model
    model.fit(train_generator,
              steps_per_epoch=len(train_images) // 32,  # assuming batch_size of 32
              validation_data=(val_images, val_masks),
              epochs=epochs)

# Example usage
input_shape = (256, 256, 3)  # Example input shape
num_classes = 3  # Number of segmentation classes
model_instance = DeepLearningModels('unet', input_shape, num_classes)

# Load and preprocess data (placeholders)
train_images, train_masks, val_images, val_masks = load_data()

# Train the model
train_unet(model_instance.model, train_images, train_masks, val_images, val_masks)

