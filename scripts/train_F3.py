import sys
import importlib
from os import path
# Add a directory to the search path
sys.path.append('/Users/vanderhoeffalex/Library/CloudStorage/OneDrive-TheBostonConsultingGroup,Inc/Desktop/Seismic_AI')
from utils.utils import (
    preprocess_traces,
    find_max_trace_length,
    read_segy_file,
    visualize_seismic_data,
    train_contrastive_ts,
    segment_into_sequences,
    create_patches,
    scale_subsequences,
    reshape_into_subsequences,
    generate_test_pairs,
    plot_aline
)
from utils.metrics import (
    pixelwise_accuracy,
    class_accuracy,
    mean_class_accuracy,
    mean_iou,
    
    
)
from data.preprocess import TimeSeriesTFRecordReader,ImageMaskTFRecordWriter,TimeSeriesTFRecordWriter,TFRecordReader,TFRecordWriter
from models.contrastive_learning import ContrastiveTimeSeriesModel, TimeSeriesAugmentation, Contrastive1DCNNModel
from models.contrastive_learning import generate_pairs, contrastive_loss, augment_time_series

import numpy as np
import tensorflow as tf
import pandas as pd
import keras as ks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.losses import SparseCategoricalCrossentropy

# Create a loss function
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
num_classes=5
# Assuming these are already defined
pixel_accuracy = pixelwise_accuracy(num_classes)
mean_cls_acc = mean_class_accuracy(num_classes)
m_iou = mean_iou(num_classes)

@tf.function
def train_step(imgs, masks):
    with tf.GradientTape() as tape:
        predictions = model(imgs, training=True)
        loss = loss_fn(masks, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update metrics
    pixel_accuracy.update_state(masks, predictions)
    mean_cls_acc.update_state(masks, predictions)
    m_iou.update_state(masks, predictions)

    return loss

def train(dataset, epochs):
    for epoch in range(epochs):
        # Reset metrics at the start of each epoch
        pixel_accuracy.reset_states()
        mean_cls_acc.reset_states()
        m_iou.reset_states()

        for batch, (imgs, masks) in enumerate(dataset):
            loss = train_step(imgs, masks)
            if batch % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch}, Loss: {loss.numpy()}')

        print(f"Epoch {epoch+1} completed. Metrics:")
        print(f"Pixel Accuracy: {pixel_accuracy.result().numpy()}")
        print(f"Mean Class Accuracy: {mean_cls_acc.result().numpy()}")
        print(f"Mean IOU: {m_iou.result().numpy()}")
