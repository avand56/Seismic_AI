import tensorflow as tf
import numpy as np

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
        image = np.load(image_path.numpy(), allow_pickle=True).astype(np.float32)
        mask = np.load(mask_path.numpy(), allow_pickle=True).astype(np.float32)

        # Apply augmentations
        if self.augment:
            image, mask = self.augment(image, mask)

        return image, mask

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
