import tensorflow as tf
import os
import numpy as np


class TFRecordWriter:
    def __init__(self, filename):
        self.filename = filename

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def write_images_to_tfr(self, images, labels=None):
        """
        Write image data and optionally labels to a TFRecord file.

        Args:
        - images (np.array): Numpy array of 2D or 3D images.
        - labels (np.array): Optional, numpy array of labels corresponding to the images.
        """
        with tf.io.TFRecordWriter(self.filename) as writer:
            for i, img in enumerate(images):
                # Ensure image is in byte format
                img_bytes = tf.io.serialize_tensor(img).numpy()
                features = {
                    'image': self._bytes_feature(img_bytes)
                }
                if labels is not None:
                    label = labels[i]
                    features['label'] = self._int64_feature(label)

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())



class TFRecordReader:
    def __init__(self, filename):
        self.filename = filename

    def _parse_image_function(self, example_proto):
        """
        Parse images and labels from the TFRecord file.
        """
        # Define your parsing schema
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        return tf.io.parse_single_example(example_proto, image_feature_description)

    def decode_image(self, features):
        """
        Decode images from the TFRecord file.
        """
        img = tf.io.parse_tensor(features['image'], out_type=tf.float32)
        label = features['label']
        return img, label

    def load_dataset(self):
        """
        Load dataset from the TFRecord file.
        """
        raw_dataset = tf.data.TFRecordDataset(self.filename)
        parsed_dataset = raw_dataset.map(self._parse_image_function)
        return parsed_dataset.map(self.decode_image)


class TimeSeriesTFRecordWriter:
    def __init__(self, filename):
        self.filename = filename

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def write_series_to_tfr(self, series_data, labels=None):
        """
        Write 1D time series data and optionally labels to a TFRecord file.

        Args:
        - series_data (np.array): Numpy array of 1D time series.
        - labels (np.array): Optional, numpy array of labels corresponding to the series.
        """
        with tf.io.TFRecordWriter(self.filename) as writer:
            for i, series in enumerate(series_data):
                # Ensure series is in byte format
                series_bytes = tf.io.serialize_tensor(series).numpy()
                features = {
                    'series': self._bytes_feature(series_bytes)
                }
                if labels is not None:
                    label = labels[i]
                    features['label'] = self._int64_feature(label)

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())

class TimeSeriesTFRecordReader:
    def __init__(self, filename):
        self.filename = filename

    def _parse_series_function(self, example_proto):
        """
        Parse series and labels from the TFRecord file.
        """
        series_feature_description = {
            'series': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        return tf.io.parse_single_example(example_proto, series_feature_description)

    def decode_series(self, features):
        """
        Decode series from the TFRecord file.
        """
        series = tf.io.parse_tensor(features['series'], out_type=tf.float32)
        label = features['label']
        return series, label

    def load_dataset(self):
        """
        Load dataset from the TFRecord file.
        """
        raw_dataset = tf.data.TFRecordDataset(self.filename)
        parsed_dataset = raw_dataset.map(self._parse_series_function)
        return parsed_dataset.map(self.decode_series)

class ImageMaskTFRecordWriter:
    def __init__(self, filename):
        self.filename = filename

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_images_and_masks_to_tfr(self, images, masks):
        """
        Write image data and corresponding masks to a TFRecord file.

        Args:
        - images (np.array): Numpy array of images.
        - masks (np.array): Numpy array of masks corresponding to the images.
        """
        with tf.io.TFRecordWriter(self.filename) as writer:
            for img, mask in zip(images, masks):
                img_bytes = tf.io.serialize_tensor(img).numpy()
                mask_bytes = tf.io.serialize_tensor(mask).numpy()
                
                features = tf.train.Features(feature={
                    'image': self._bytes_feature(img_bytes),
                    'mask': self._bytes_feature(mask_bytes),
                })
                tf_example = tf.train.Example(features=features)
                writer.write(tf_example.SerializeToString())

        def parse_tfrecord_fn(example_proto):
            """
            Parses TFRecord entries to TensorFlow representations.
            """
            feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'mask': tf.io.FixedLenFeature([], tf.string),
            }
            example = tf.io.parse_single_example(example_proto, feature_description)
            image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
            mask = tf.io.parse_tensor(example['mask'], out_type=tf.float32)
            return image, mask

        def load_dataset_from_tfrecords(tfrecord_filename):
            """
            Loads a dataset of images and masks from TFRecord files.
            """
            raw_dataset = tf.data.TFRecordDataset([tfrecord_filename])
            parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
            return parsed_dataset


# train_dataset = load_dataset_from_tfrecords('train.tfrecords')
# val_dataset = load_dataset_from_tfrecords('val.tfrecords')

# # Further processing such as batching, prefetching, augmentations can be applied to these datasets.
# train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
# val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


# Writing Example
# images = np.random.rand(10, 224, 224, 3).astype(np.float32)  # Example 2D images
# labels = np.random.randint(0, 2, size=(10,)).astype(np.int64)  # Example labels
# writer = TFRecordWriter('images.tfrecords')
# writer.write_images_to_tfr(images, labels)

# # Reading Example
# reader = TFRecordReader('images.tfrecords')
# dataset = reader.load_dataset()

# for img, label in dataset.take(1):
#     print(img.shape, label.numpy())
    

# Example time series data
# series_data = np.random.rand(100, 128).astype(np.float32)  # 100 time series, each 128 steps long
# labels = np.random.randint(0, 2, size=(100,)).astype(np.int64)  # Example binary labels for each series

# # Writing to TFRecord
# writer = TimeSeriesTFRecordWriter('time_series.tfrecords')
# writer.write_series_to_tfr(series_data, labels)

# # Reading from TFRecord
# reader = TimeSeriesTFRecordReader('time_series.tfrecords')
# dataset = reader.load_dataset()

# for series, label in dataset.take(1):
#     print(series.numpy().shape, label.numpy())
