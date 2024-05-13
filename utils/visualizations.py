
import matplotlib.pyplot as plt
import numpy as np
from utils.patch import reconstruct_image

def plot_slices(volume, num_slices=6):
    """ Plot horizontal slices of the 3D volume. """
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    slice_positions = np.linspace(0, volume.shape[2] - 1, num_slices, dtype=np.int)
    for i, ax in enumerate(axes):
        ax.imshow(volume[:, :, slice_positions[i]], cmap='viridis')
        ax.set_title(f'Slice {slice_positions[i]}')
        ax.axis('off')
    plt.show()


def visualize_original_vs_prediction(original_patches, prediction_patches, original_dims):
    # Reconstruct full images from patches
    original_image = reconstruct_image(original_patches, original_dims)
    predicted_image = reconstruct_image(prediction_patches, original_dims)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Predicted Image')
    plt.imshow(predicted_image, cmap='rgb')
    plt.axis('off')

    plt.show()

def reconstruct_from_patches(patches, original_shape, patch_size, stride):
    """
    Reconstruct a 3D volume from 2D patches assuming each patch corresponds to a slice in depth.
    
    Args:
    patches (numpy.ndarray): The array of patches of shape (num_patches, patch_height, patch_width).
    original_shape (tuple): The shape of the full volume as (height, width, depth).
    patch_size (tuple): The height and width of each patch as (patch_height, patch_width).
    stride (tuple): The stride used to slide the patch extraction window as (stride_height, stride_width).
    
    Returns:
    numpy.ndarray: The reconstructed 3D volume.
    """
    # Initialize the reconstructed volume
    reconstructed = np.zeros(original_shape)
    
    # Calculate the number of patches along each axis
    num_patches_height = (original_shape[0] - patch_size[0]) // stride[0] + 1
    num_patches_width = (original_shape[1] - patch_size[1]) // stride[1] + 1
    num_depth_slices = original_shape[2]
    
    # Assume each depth slice is processed independently and patches are arranged sequentially in depth
    patch_idx = 0
    for z in range(num_depth_slices):
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                if patch_idx < len(patches):
                    start_i = i * stride[0]
                    start_j = j * stride[1]
                    reconstructed[start_i:start_i + patch_size[0], start_j:start_j + patch_size[1], z] = patches[patch_idx]
                    patch_idx += 1

    return reconstructed

def plot_original_and_reconstructed(original, reconstructed, depth_slice):
    """
    Plot the original and reconstructed data side by side for a given depth slice.
    
    Args:
    original (numpy.ndarray): The original 3D seismic volume.
    reconstructed (numpy.ndarray): The reconstructed 3D volume from patches.
    depth_slice (int): The specific depth slice to visualize.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Data')
    plt.imshow(original[:, :, depth_slice], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Predictions')
    plt.imshow(reconstructed[:, :, depth_slice], cmap='grey')
    plt.axis('off')

    plt.show()


def plot_sections(original, reconstructed, inline=None, crossline=None):
    """
    Plot the original and reconstructed data side by side for a given inline or crossline slice.
    
    Args:
    original (numpy.ndarray): The original 3D seismic volume.
    reconstructed (numpy.ndarray): The reconstructed 3D volume from patches.
    inline (int): The specific inline slice to visualize. If None, crossline must be provided.
    crossline (int): The specific crossline slice to visualize. If None, inline must be provided.
    """
    plt.figure(figsize=(12, 6))
    
    if inline is not None:
        # Plotting inline slices
        original_slice = original[:, inline, :]
        reconstructed_slice = reconstructed[:, inline, :]
        title = f'Inline {inline}'
    elif crossline is not None:
        # Plotting crossline slices
        original_slice = original[crossline, :, :]
        reconstructed_slice = reconstructed[crossline, :, :]
        title = f'Crossline {crossline}'
    else:
        raise ValueError("Either inline or crossline index must be provided.")

    plt.subplot(1, 2, 1)
    plt.title(f'Original {title}')
    plt.imshow(original_slice.T, cmap='gray', aspect='auto')  # Transposed for correct orientation
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Reconstructed {title}')
    plt.imshow(reconstructed_slice.T, cmap='gray', aspect='auto')  # Transposed for correct orientation
    plt.axis('off')

    plt.show()