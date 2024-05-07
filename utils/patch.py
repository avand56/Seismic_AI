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
class PatchLoader:
    def __init__(self, image_paths, mask_paths, patch_size=(64, 64), batch_size=32, augment=None):
        assert len(image_paths) == len(mask_paths), "Images and masks must have the same number of items."
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.augment = augment
        self.dataset = self.create_dataset()

    def load_and_augment(self, image_path, mask_path):
        # Load data
        images = np.load(image_path.numpy(), allow_pickle=True).astype(np.float32)
        masks = np.load(mask_path.numpy(), allow_pickle=True).astype(np.float32)

        # Apply augmentations
        if self.augment:
            images, masks = self.augment(images, masks)

        return images, masks

    def extract_patches(self, image, mask):
        image = tf.expand_dims(image, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

        image_patches = tf.image.extract_patches(
            images=tf.expand_dims(image, 0),
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding='VALID')

        mask_patches = tf.image.extract_patches(
            images=tf.expand_dims(mask, 0),
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding='VALID')

        return tf.reshape(image_patches, [-1, self.patch_size[0], self.patch_size[1], 1]), \
               tf.reshape(mask_patches, [-1, self.patch_size[0], self.patch_size[1], 1])

    def create_dataset(self):
        # Create a dataset from file paths
        path_ds = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        ds = path_ds.map(lambda x, y: tf.py_function(self.load_and_augment, [x, y], [tf.float32, tf.float32]),
                         num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda img, mask: self.extract_patches(img, mask)).unbatch()

        # Shuffle and batch the data
        ds = ds.shuffle(1000).batch(self.batch_size)
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

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


class SeismicDataPreprocessor:
    def __init__(self, patch_size, stride, augmentations, num_augmentations=2, test_size=0.2):
        self.patch_size = patch_size
        self.stride = stride
        self.augmentations = augmentations
        self.num_augmentations = num_augmentations
        self.test_size = test_size

    def extract_and_split_patches(self, images, masks):
        dataset, patches = self.extract_patches(images, masks)
        # Splitting the dataset into training and validation sets
        train_data, val_data = train_test_split(patches, test_size=self.test_size, random_state=42)

        # Convert lists back to tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).map(self.split_image_mask)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data).map(self.split_image_mask)

        return train_dataset, val_dataset

    def extract_patches(self, data, masks):
        height, width, depth = data.shape
        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.stride
        
        patches = []
        for z in range(depth):
            for i in range(0, height - patch_height + 1, stride_height):
                for j in range(0, width - patch_width + 1, stride_width):
                    img_patch = data[z, i:i + patch_height, j:j + patch_width]
                    mask_patch = masks[z, i:i + patch_height, j:j + patch_width]
                    for _ in range(self.num_augmentations):
                        img_aug, mask_aug = self.apply_augmentations(img_patch, mask_patch)
                        patches.append((img_aug[..., np.newaxis], mask_aug[..., np.newaxis]))

        return tf.data.Dataset.from_tensor_slices(patches), patches

    def apply_augmentations(self, img_patch, mask_patch):
        img_patch = Image.fromarray(img_patch, mode='L')
        mask_patch = Image.fromarray(mask_patch, mode='L')
        for a in self.augmentations:
            img_patch, mask_patch = a(img_patch, mask_patch)
        return np.array(img_patch), np.array(mask_patch)

    def split_image_mask(self, combined_tensor):
        image = combined_tensor[0]  # First part is the image
        mask = combined_tensor[1]  # Second part is the mask
        return image, mask

class SeismicDataPreprocessor1D:
    def __init__(self, augmentations, test_size=0.2):
        """
        Initializes the data processor for seismic data with data augmentation for 1D traces.
        
        Args:
        - augmentations (list): List of augmentation transformations to apply to 1D traces.
        - test_size (float): The fraction of the dataset to reserve for validation.
        """
        self.augmentations = augmentations
        self.test_size = test_size

    def extract_and_split_traces(self, images, masks):
        dataset, traces = self.extract_traces(images, masks)
        # Splitting the dataset into training and validation sets
        train_data, val_data = train_test_split(traces, test_size=self.test_size, random_state=42)

        # Convert lists back to tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).map(self.split_image_mask)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data).map(self.split_image_mask)

        return train_dataset, val_dataset

    def extract_traces(self, data, masks):
        depth, height, width = data.shape
        traces = []
        
        # Extracting inline traces
        for i in range(height):
            for d in range(depth):
                img_trace = data[d, i, :]
                mask_trace = masks[d, i, :]
                img_trace, mask_trace = self.apply_augmentations(img_trace, mask_trace)
                traces.append((img_trace, mask_trace))
        
        # Extracting crossline traces
        for j in range(width):
            for d in range(depth):
                img_trace = data[d, :, j]
                mask_trace = masks[d, :, j]
                img_trace, mask_trace = self.apply_augmentations(img_trace, mask_trace)
                traces.append((img_trace, mask_trace))

        return tf.data.Dataset.from_tensor_slices(traces), traces

    def apply_augmentations(self, img_trace, mask_trace):
        # Apply any 1D or appropriate augmentations here
        # This is a placeholder to include your actual augmentation logic
        return img_trace, mask_trace

    def split_image_mask(self, combined_tensor):
        image = combined_tensor[0]  # First part is the image trace
        mask = combined_tensor[1]  # Second part is the mask trace
        return image, mask

class SeismicDataPreprocessor3D:
    def __init__(self, cube_size, stride, augmentations, test_size=0.2):
        """
        Initializes the data processor for seismic data with data augmentation for 3D cubes.
        
        Args:
        - cube_size (tuple): The dimensions (depth, height, width) of the cubes to extract.
        - stride (tuple): The strides (step in depth, step in height, step in width) for the sliding window.
        - augmentations (list): List of augmentation transformations to apply.
        - test_size (float): The fraction of the dataset to reserve for validation.
        """
        self.cube_size = cube_size
        self.stride = stride
        self.augmentations = augmentations
        self.test_size = test_size

    def extract_and_split_cubes(self, images, masks):
        dataset, patches = self.extract_cubes(images, masks)
        # Splitting the dataset into training and validation sets
        train_data, val_data = train_test_split(patches, test_size=self.test_size, random_state=42)

        # Convert lists back to tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).map(self.split_image_mask)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data).map(self.split_image_mask)

        return train_dataset, val_dataset

    def extract_cubes(self, data, masks):
        depth, height, width = data.shape
        cube_depth, cube_height, cube_width = self.cube_size
        stride_depth, stride_height, stride_width = self.stride
        
        patches = []
        for z in range(0, depth - cube_depth + 1, stride_depth):
            for i in range(0, height - cube_height + 1, stride_height):
                for j in range(0, width - cube_width + 1, stride_width):
                    img_cube = data[z:z + cube_depth, i:i + cube_height, j:j + cube_width]
                    mask_cube = masks[z:z + cube_depth, i:i + cube_height, j:j + cube_width]
                    img_cube, mask_cube = self.apply_augmentations(img_cube, mask_cube)
                    # Ensure the cubes are in the format TensorFlow expects (extra dimension for channels)
                    patches.append((img_cube[..., np.newaxis], mask_cube[..., np.newaxis]))

        return tf.data.Dataset.from_tensor_slices(patches), patches

    def apply_augmentations(self, img_cube, mask_cube):
        # Apply any 3D or appropriate augmentations here
        # This is a placeholder to include your actual augmentation logic
        return img_cube, mask_cube

    def split_image_mask(self, combined_tensor):
        image = combined_tensor[0]  # First part is the image cube
        mask = combined_tensor[1]  # Second part is the mask cube
        return image, mask
    
# class SeismicDataPreprocessor:
#     def __init__(self, patch_size, stride, augmentations, test_size=0.2):
#         """
#         Initializes the data processor for seismic data with data augmentation.
        
#         Args:
#         - patch_size (tuple): The dimensions (height, width) of the patches to extract.
#         - stride (tuple): The strides (step in height, step in width) for the sliding window.
#         - augmentations (list): List of augmentation transformations to apply.
#         - test_size (float): The fraction of the dataset to reserve for validation.
#         """
#         self.patch_size = patch_size
#         self.stride = stride
#         self.augmentations = augmentations
#         self.test_size = test_size

#     def extract_and_split_patches(self, images, masks):
#         """
#         Extracts patches from the images and masks, applies augmentations, and splits into training and validation datasets.
        
#         Args:
#         - images (np.array): 3D array of images, shape (depth, height, width).
#         - masks (np.array): 3D array of masks, shape (depth, height, width).
        
#         Returns:
#         - train_dataset (tf.data.Dataset): Training dataset containing (image_patch, mask_patch) tuples.
#         - val_dataset (tf.data.Dataset): Validation dataset containing (image_patch, mask_patch) tuples.
#         """
#         dataset, patches = self.extract_patches(images, masks)
#         # Splitting the dataset into training and validation sets
#         train_data, val_data = train_test_split(list(dataset), test_size=self.test_size, random_state=42)

#         # Convert lists back to tf.data.Dataset
#         train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
#         val_dataset = tf.data.Dataset.from_tensor_slices(val_data)

#         return train_dataset, val_dataset,train_data, val_data
    

#     def extract_patches(self, data, masks):
#         """
#         Extracts patches from data and applies augmentations.
#         Combines functionality of the earlier provided extract_patches and data augmentation logic.
#         """
#         depth, height, width = data.shape
#         patch_height, patch_width = self.patch_size
#         stride_height, stride_width = self.stride
        
#         patches = []
#         for z in range(depth):
#             for i in range(0, height - patch_height + 1, stride_height):
#                 for j in range(0, width - patch_width + 1, stride_width):
#                     img_patch = data[z, i:i + patch_height, j:j + patch_width]
#                     mask_patch = masks[z, i:i + patch_height, j:j + patch_width]
#                     img_patch, mask_patch = self.apply_augmentations(img_patch, mask_patch)
#                     patches.append((img_patch[..., np.newaxis], mask_patch[..., np.newaxis]))
        
#         return tf.data.Dataset.from_tensor_slices(patches), patches

#     def apply_augmentations(self, img_patch, mask_patch):
#         """
#         Apply configured augmentations to the image and mask patches.
#         """
#         img_patch = Image.fromarray(img_patch, mode='L')
#         mask_patch = Image.fromarray(mask_patch, mode='L')
#         for a in self.augmentations:
#             img_patch, mask_patch = a(img_patch, mask_patch)
#         return np.array(img_patch), np.array(mask_patch)
    
#     def split_image_mask(self,combined_tensor):
#         """
#         Splits the combined image-mask tensor into separate image and mask tensors.

#         Args:
#         combined_tensor (tf.Tensor): Input tensor of shape (2, 128, 128, 1)
#                                     where the first slice [0] is the image and the second slice [1] is the mask.

#         Returns:
#         image (tf.Tensor): The image tensor of shape (128, 128, 1).
#         mask (tf.Tensor): The mask tensor of shape (128, 128, 1).
#         """
#         image = combined_tensor[0]  # First part is the image
#         mask = combined_tensor[1]  # Second part is the mask
#         return image, mask

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
