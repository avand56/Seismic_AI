import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import MeanIoU
import numpy as np
from models.model_class import DeepLearningModels


def train_and_evaluate_model(model, train_dataset, val_dataset, epochs=100, batch_size=32, learning_rate=1e-4, model_save_path='best_model.h5', loss='categorical_crossentropy', metrics=['accuracy'], additional_callbacks=[]):
    # check if GPU recognized.
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
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
