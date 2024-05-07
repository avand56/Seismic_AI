import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import random
import math
from utils.augmentations import (
    DataAugmentationImage,
    Compose,
    AddNoise,
    RandomCrop,
    CenterCrop,
    RandomHorizontallyFlip,
    RandomVerticallyFlip,
    FreeScale,Scale,
    RandomSizedCrop,
    RandomRotate,
    RandomSized
)

# Patch Loader class definition that handles both images and masks
# class PatchLoader:
#     def __init__(self, image_paths, mask_paths, patch_size=(64, 64), batch_size=32, augment=None):
#         assert len(image_paths) == len(mask_paths), "Images and masks must have the same number of items."
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.patch_size = patch_size
#         self.batch_size = batch_size
#         self.augment = augment
#         self.dataset = self.create_dataset()

#     def load_and_augment(self, image_path, mask_path):
#         # Load data
#         images = np.load(image_path.numpy(), allow_pickle=True).astype(np.float32)
#         masks = np.load(mask_path.numpy(), allow_pickle=True).astype(np.float32)

#         # Apply augmentations
#         if self.augment:
#             images, masks = self.augment(images, masks)

#         return images, masks

#     def extract_patches(self, image, mask):
#         image = tf.expand_dims(image, axis=-1)
#         mask = tf.expand_dims(mask, axis=-1)

#         image_patches = tf.image.extract_patches(
#             images=tf.expand_dims(image, 0),
#             sizes=[1, self.patch_size[0], self.patch_size[1], 1],
#             strides=[1, self.patch_size[0], self.patch_size[1], 1],
#             rates=[1, 1, 1, 1],
#             padding='VALID')

#         mask_patches = tf.image.extract_patches(
#             images=tf.expand_dims(mask, 0),
#             sizes=[1, self.patch_size[0], self.patch_size[1], 1],
#             strides=[1, self.patch_size[0], self.patch_size[1], 1],
#             rates=[1, 1, 1, 1],
#             padding='VALID')

#         return tf.reshape(image_patches, [-1, self.patch_size[0], self.patch_size[1], 1]), \
#                tf.reshape(mask_patches, [-1, self.patch_size[0], self.patch_size[1], 1])

#     def create_dataset(self):
#         # Create a dataset from file paths
#         path_ds = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
#         ds = path_ds.map(lambda x, y: tf.py_function(self.load_and_augment, [x, y], [tf.float32, tf.float32]),
#                          num_parallel_calls=tf.data.AUTOTUNE)
#         ds = ds.map(lambda img, mask: self.extract_patches(img, mask)).unbatch()

#         # Shuffle and batch the data
#         ds = ds.shuffle(1000).batch(self.batch_size)
#         return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

class SeismicProcessor:
    def __init__(self, patch_size, stride, augmentations, num_augmentations=2, test_size=0.2):
        self.patch_size = patch_size  # Tuple (height, width)
        self.stride = stride  # Tuple (height_stride, width_stride)
        self.augmentations = augmentations
        self.num_augmentations = num_augmentations
        self.test_size = test_size
        
    def extract_patches(self, volume):
        # volume shape assumed to be (crossline, inline, depth)
        # Adjusting logic to slice along depth and extract (128, 128) patches from crossline and inline dimensions
        patches = []
        depth_slices = range(volume.shape[2])  # Iterate through each depth slice
        for z in depth_slices:
            for i in range(0, volume.shape[0] - self.patch_size[0] + 1, self.stride[0]):
                for j in range(0, volume.shape[1] - self.patch_size[1] + 1, self.stride[1]):
                    patch = volume[i:i + self.patch_size[0], j:j + self.patch_size[1], z]
                    patch = np.expand_dims(patch, axis=-1)  # Ensure it's a 3D array (128, 128, 1)
                    patches.append(patch)
        return np.array(patches)

    def apply_augmentations(self, image_patches, label_patches):
        augmented_images = []
        augmented_labels = []
        for image, label in zip(image_patches, label_patches):
            for _ in range(self.num_augmentations):
                for func in self.augmentations:
                    aug_image, aug_label = func(image, label)
                    augmented_images.append(aug_image)
                    augmented_labels.append(aug_label)
        return np.array(augmented_images), np.array(augmented_labels)

    def create_datasets(self, seismic_data, labels):
        # Extract patches
        image_patches = self.extract_patches(seismic_data)
        label_patches = self.extract_patches(labels)
        
        # Apply augmentations
        images, labels = self.apply_augmentations(image_patches, label_patches)
        
        # Squeeze unnecessary dimensions using NumPy
        # images = np.squeeze(images, axis=-1)  # Ensure no unwanted singleton dimensions for images
        # labels = np.squeeze(labels, axis=-1) 

        # Split the dataset
        train_images, val_images, train_labels, val_labels = train_test_split(
            images, labels, test_size=self.test_size, random_state=42)
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        
        return train_dataset, val_dataset,train_images,train_labels


# class SeismicDataPreprocessor:
#     def __init__(self, patch_size, stride, augmentations, num_augmentations=2, test_size=0.2):
#         self.patch_size = patch_size
#         self.stride = stride
#         self.augmentations = augmentations
#         self.num_augmentations = num_augmentations
#         self.test_size = test_size

#     def extract_and_split_patches(self, images, masks):
#         dataset, patches = self.extract_patches(images, masks)
#         # Splitting the dataset into training and validation sets
#         train_data, val_data = train_test_split(patches, test_size=self.test_size, random_state=42)

#         # Convert lists back to tf.data.Dataset
#         train_dataset = tf.data.Dataset.from_tensor_slices(train_data).map(self.split_image_mask)
#         val_dataset = tf.data.Dataset.from_tensor_slices(val_data).map(self.split_image_mask)

#         return train_dataset, val_dataset

#     def extract_patches(self, data, masks):
#         height, width, depth = data.shape
#         patch_height, patch_width = self.patch_size
#         stride_height, stride_width = self.stride
        
#         patches = []
#         for z in range(depth):
#             for i in range(0, height - patch_height + 1, stride_height):
#                 for j in range(0, width - patch_width + 1, stride_width):
#                     img_patch = data[z, i:i + patch_height, j:j + patch_width]
#                     mask_patch = masks[z, i:i + patch_height, j:j + patch_width]
#                     for _ in range(self.num_augmentations):
#                         img_aug, mask_aug = self.apply_augmentations(img_patch, mask_patch)
#                         patches.append((img_aug[..., np.newaxis], mask_aug[..., np.newaxis]))

#         return tf.data.Dataset.from_tensor_slices(patches), patches

#     def apply_augmentations(self, img_patch, mask_patch):
#         img_patch = Image.fromarray(img_patch, mode='L')
#         mask_patch = Image.fromarray(mask_patch, mode='L')
#         for a in self.augmentations:
#             img_patch, mask_patch = a(img_patch, mask_patch)
#         return np.array(img_patch), np.array(mask_patch)

#     def split_image_mask(self, combined_tensor):
#         image = combined_tensor[0]  # First part is the image
#         mask = combined_tensor[1]  # Second part is the mask
#         return image, mask

class SeismicProcessor1D:
    def __init__(self, num_traces, trace_length, augmentations, num_augmentations=2, test_size=0.2):
        """
        Initialize the SeismicProcessor for 1D trace extraction.
        :param num_traces: Integer, the number of traces to extract.
        :param trace_length: Integer, the number of depth points in each trace.
        :param augmentations: List of functions to augment the extracted traces.
        :param num_augmentations: Integer, number of times to augment each trace.
        :param test_size: Float, fraction of data to use as validation set.
        """
        self.num_traces = num_traces
        self.trace_length = trace_length
        self.augmentations = augmentations
        self.num_augmentations = num_augmentations
        self.test_size = test_size

    def extract_traces(self, volume):
        """
        Extract 1D traces from a 3D seismic volume.
        :param volume: 3D numpy array from which to extract traces.
        :return: numpy array of extracted traces.
        """
        traces = []
        depth, height, width = volume.shape
        selected_indices = np.random.choice(height * width, self.num_traces, replace=False)
        
        for idx in selected_indices:
            h_idx = idx // width
            w_idx = idx % width
            if self.trace_length and depth >= self.trace_length:
                start_depth = np.random.randint(0, depth - self.trace_length)
                trace = volume[start_depth:start_depth + self.trace_length, h_idx, w_idx]
            else:
                trace = volume[:, h_idx, w_idx]
            traces.append(trace)

        return np.array(traces)

    def apply_augmentations(self, traces):
        """
        Apply augmentations to the extracted traces.
        :param traces: numpy array of traces to augment.
        :return: numpy array of augmented traces.
        """
        augmented_traces = []
        for trace in traces:
            for _ in range(self.num_augmentations):
                for func in self.augmentations:
                    trace = func(trace)
                augmented_traces.append(trace)
        return np.array(augmented_traces)

    def create_datasets(self, seismic_data):
        """
        Extract traces, apply augmentations, and create training and validation datasets.
        :param seismic_data: 3D numpy array of seismic data.
        :return: TensorFlow datasets for training and validation.
        """
        traces = self.extract_traces(seismic_data)
        traces = self.apply_augmentations(traces)
        
        train_traces, val_traces = train_test_split(traces, test_size=self.test_size, random_state=42)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(train_traces)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_traces)
        
        return train_dataset, val_dataset

class SeismicProcessor3D:
    def __init__(self, cube_size, stride, augmentations, num_augmentations=2, test_size=0.2):
        """
        Initialize the SeismicProcessor with parameters for extracting 3D cubes.
        :param cube_size: Tuple of three integers (depth, height, width) for the size of the cubes.
        :param stride: Tuple of three integers (depth_stride, height_stride, width_stride) for the stride of the extraction.
        :param augmentations: List of functions to augment the extracted cubes.
        :param num_augmentations: Number of times to augment each cube.
        :param test_size: Fraction of data to use as validation set.
        """
        self.cube_size = cube_size
        self.stride = stride
        self.augmentations = augmentations
        self.num_augmentations = num_augmentations
        self.test_size = test_size

    def extract_cubes(self, volume):
        """
        Extract 3D cubes from a 3D seismic volume.
        :param volume: 3D numpy array from which to extract cubes.
        :return: numpy array of extracted cubes.
        """
        depth, height, width = volume.shape
        cubes = []
        for d in range(0, depth - self.cube_size[0] + 1, self.stride[0]):
            for i in range(0, height - self.cube_size[1] + 1, self.stride[1]):
                for j in range(0, width - self.cube_size[2] + 1, self.stride[2]):
                    cube = volume[d:d + self.cube_size[0], i:i + self.cube_size[1], j:j + self.cube_size[2]]
                    cubes.append(cube)
        return np.array(cubes)

    def apply_augmentations(self, cubes):
        """
        Apply augmentations to the extracted cubes.
        :param cubes: numpy array of cubes to augment.
        :return: numpy array of augmented cubes.
        """
        augmented_cubes = []
        for cube in cubes:
            for _ in range(self.num_augmentations):
                for func in self.augmentations:
                    cube = func(cube)
                augmented_cubes.append(cube)
        return np.array(augmented_cubes)

    def create_datasets(self, seismic_data):
        """
        Extract cubes, apply augmentations, and create training and validation datasets.
        :param seismic_data: 3D numpy array of seismic data.
        :return: TensorFlow datasets for training and validation.
        """
        cubes = self.extract_cubes(seismic_data)
        cubes = self.apply_augmentations(cubes)
        
        train_cubes, val_cubes = train_test_split(cubes, test_size=self.test_size, random_state=42)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(train_cubes)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_cubes)
        
        return train_dataset, val_dataset
#########
# To run
#######
    

# Example usage
# augmentations = [RandomHorizontallyFlip()]  # Define more augmentations as needed
# augmenter = DataAugmentation(augmentations)
# image_paths = ['path_to_image1.npy', 'path_to_image2.npy']  # Paths to your image data files
# mask_paths = ['path_to_mask1.npy', 'path_to_mask2.npy']  # Paths to your mask data files
# patch_loader = PatchLoader(image_paths, mask_paths, patch_size=(128, 128), batch_size=16, augment=augmenter)

# # Accessing the dataset
# dataset = patch_loader.dataset
# for images, masks in dataset.take(1):
#     print(images.shape, masks.shape)  # Check the shapes
