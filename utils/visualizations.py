import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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


def visualize_slices(direction, original, reconstructed, initial_slice=0):
    # Determine the number of slices based on the direction
    num_slices = original.shape[{'inline': 0, 'crossline': 1, 'depth': 2}[direction]]

    # Create a blank figure with the initial slices
    fig = go.Figure()

    # Function to extract a slice from the volume
    def get_slice(data, index):
        if direction == 'inline':
            return data[index, :, :]
        elif direction == 'crossline':
            return data[:, index, :]
        else:  # depth
            return data[:, :, index]

    # Add the initial visible traces for the original and reconstructed slices
    fig.add_trace(go.Image(z=get_slice(original, initial_slice), name="Original"))
    fig.add_trace(go.Image(z=get_slice(reconstructed, initial_slice), name="Reconstructed"))

    # Create steps for the slider
    steps = []
    for i in range(num_slices):
        step = dict(
            method="update",
            args=[{"z": [get_slice(original, i), get_slice(reconstructed, i)]},
                  {"title": f"{direction.capitalize()} Slice {i}"}],
            label=f'Slice {i}')
        steps.append(step)

    sliders = [dict(
        active=initial_slice,
        currentvalue={"prefix": "Current Slice: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)

    return fig


def visualize_slices1(direction, original, reconstructed):
    fig = go.Figure()

    if direction == 'inline':
        num_slices = original.shape[0]
        slice_dim = 0
    elif direction == 'crossline':
        num_slices = original.shape[1]
        slice_dim = 1
    else:  # depth
        num_slices = original.shape[2]
        slice_dim = 2

    # Add traces for all slices in the chosen direction
    for i in range(num_slices):
        if slice_dim == 0:  # inline
            original_slice = original[i, :, :]
            reconstructed_slice = reconstructed[i, :, :]
        elif slice_dim == 1:  # crossline
            original_slice = original[:, i, :]
            reconstructed_slice = reconstructed[:, i, :]
        else:  # depth
            original_slice = original[:, :, i]
            reconstructed_slice = reconstructed[:, :, i]

        fig.add_trace(
            go.Image(z=original_slice,
                     visible=(i==0),
                     name=f'Original {direction} {i}'))
        fig.add_trace(
            go.Image(z=reconstructed_slice,
                     visible=(i==0),
                     name=f'Reconstructed {direction} {i}'))

    # Create steps for the slider
    steps = []
    for i in range(num_slices):
        step = dict(
            method="update",
            args=[{"visible": [False] * (num_slices * 2)},
                  {"title": f"{direction.capitalize()} Slice: {i}"}],
            label=f'Slice {i}')
        step["args"][0]["visible"][2*i] = True  # Toggle original visibility
        step["args"][0]["visible"][2*i + 1] = True  # Toggle reconstructed visibility
        steps.append(step)

    # Create and add slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": f"Select {direction} slice: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    return fig


class vlm_slicer_interactive:
    def __init__(self, seis_vlm, pred_vlm, flag_slice):
        axis_color = 'lightgoldenrodyellow'
        
        # Setting idx_max correctly based on the flag_slice value
        if flag_slice == 0:  # Inline
            idx_max = seis_vlm.shape[0] - 1  # 601 slices
        elif flag_slice == 1:  # Crossline
            idx_max = seis_vlm.shape[1] - 1  # 200 slices
        elif flag_slice == 2:  # Depth
            idx_max = seis_vlm.shape[2] - 1  # 255 slices

        self.seis_vlm = seis_vlm
        self.pred_vlm = pred_vlm
        self.cmap_bg = plt.cm.gray_r
        self.flag_slice = flag_slice
        
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.85)
        self.ax = self.fig.add_subplot(111)
        self.idx_slider_ax = self.fig.add_axes([0.25, 0.1, 0.50, 0.03], facecolor=axis_color)
        self.idx_slider = Slider(self.idx_slider_ax, 'Slice Index', 0, idx_max, valinit=0, valfmt='%d')
        self.idx_slider.on_changed(self.sliders_on_changed)
        self.reset_button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(self.reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
        self.reset_button.on_clicked(self.reset_button_on_clicked)

        self.imshow_alpha()  # Initial display

    def sliders_on_changed(self, val):
        idx = int(val)
        if idx <= self.idx_slider.valmax:  # Ensure index is within the maximum slider range
            self.imshow_alpha(idx)

    def plot_slice(self, idx):
        """Selects the appropriate slices based on idx and flag_slice."""
        if self.flag_slice == 0:
            self.seis_slice = self.seis_vlm[idx, :, :]
            self.pred_slice = self.pred_vlm[idx, :, :]
        elif self.flag_slice == 1:
            self.seis_slice = self.seis_vlm[:, idx, :]
            self.pred_slice = self.pred_vlm[:, idx, :]
        elif self.flag_slice == 2:
            self.seis_slice = self.seis_vlm[:, :, idx]
            self.pred_slice = self.pred_vlm[:, :, idx]
            
    def create_img_alpha(self, img_input):
        """Create RGBA image with distinct colors for each class."""
        colors = np.array([
            [0, 0, 0, 0],    # Background - black, semi-transparent
            [1, 0, 0, 0.5],  # Class 1 - Red, semi-transparent
            [0, 1, 0, 0.5],  # Class 2 - Green, semi-transparent
            [0, 0, 1, 0.5],  # Class 3 - Blue, semi-transparent
            [1, 1, 0, 0.5],  # Class 4 - Yellow, semi-transparent
            [0, 1, 1, 0.5],  # Class 5 - Cyan, semi-transparent
            [1, 0, 1, 0.5]   # Class 6 - Magenta, semi-transparent
        ])
        # Allocate RGBA image based on input shape and apply colors based on class index
        img_alpha = np.zeros((*img_input.shape, 4))
        for i in range(1, 7):  # Assuming class indices start from 1
            img_alpha[img_input == i] = colors[i]
        return img_alpha

    def imshow_alpha(self, idx=0):
        self.plot_slice(idx)
        img_alpha = self.create_img_alpha(self.pred_slice.T)
        self.ax.clear()
        
        # Depending on flag_slice, set the plot axes correctly.
        if self.flag_slice == 0:  # Inline
            axis_labels = ("Crossline", "Depth")
        elif self.flag_slice == 1:  # Crossline
            axis_labels = ("Inline", "Depth")
        elif self.flag_slice == 2:  # Depth
            axis_labels = ("Inline", "Crossline")

        self.ax.imshow(self.seis_slice.T, cmap=self.cmap_bg)
        self.ax.imshow(img_alpha, alpha=0.5)
        self.ax.set_xlabel(axis_labels[0])
        self.ax.set_ylabel(axis_labels[1])
        self.ax.set_title(f"{['Inline', 'Crossline', 'Depth'][self.flag_slice]} Seismic Section {idx} and Lithofacies Classes {idx}", pad=20)

        self.fig.canvas.draw_idle()


    # def imshow_alpha(self, idx=0):
    #     """Plot seismic and prediction data side by side."""
    #     self.plot_slice(idx)
    #     img_alpha = self.create_img_alpha(self.pred_slice.T)
    #     self.ax.clear()  # Clear previous images
    #     # Display seismic data
    #     self.ax.imshow(self.seis_slice.T, cmap=self.cmap_bg, extent=[0, self.seis_slice.shape[1], 0, self.seis_slice.shape[0]])
    #     # Display prediction data right next to it
    #     offset = self.seis_slice.shape[1]
    #     self.ax.imshow(img_alpha, alpha=0.5, extent=[offset, offset + self.pred_slice.shape[1], 0, self.pred_slice.shape[0]])
    #     self.ax.set_xlim([0, offset + self.pred_slice.shape[1]])  # Adjust x-axis limits

    # def imshow_alpha(self, idx=0):
    #     """Plots the seismic and prediction data with transparency."""
    #     self.plot_slice(idx)
    #     img_alpha = self.create_img_alpha(self.pred_slice.T)
    #     self.ax.clear()  # Clear previous images
    #     self.ax.imshow(self.seis_slice.T, cmap=self.cmap_bg)
    #     self.ax.imshow(img_alpha, alpha=0.5)

    def reset_button_on_clicked(self, event):
        """Resets the slider to the initial value when the reset button is clicked."""
        self.idx_slider.reset()



class vlm_slicer_interactive2:
    def __init__(self, seis_vlm, pred_vlm, flag_slice):
        axis_color = 'lightgoldenrodyellow'
        
        # Define dimension labels here for use in other methods
        self.dimension_labels = ['Inline', 'Crossline', 'Depth']
        dimension_sizes = [seis_vlm.shape[0], seis_vlm.shape[1], seis_vlm.shape[2]]
        idx_max = dimension_sizes[flag_slice] - 1

        self.seis_vlm = seis_vlm
        self.pred_vlm = pred_vlm
        self.cmap_bg = plt.cm.gray_r
        self.flag_slice = flag_slice
        
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.85)
        self.ax = self.fig.add_subplot(111)
        self.idx_slider_ax = self.fig.add_axes([0.25, 0.1, 0.50, 0.03], facecolor=axis_color)
        self.idx_slider = Slider(self.idx_slider_ax, 'Slice Index', 0, idx_max, valinit=0, valfmt='%d')
        self.idx_slider.on_changed(self.sliders_on_changed)
        self.reset_button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(self.reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
        self.reset_button.on_clicked(self.reset_button_on_clicked)

        self.imshow_alpha()  # Initial display

    def imshow_alpha(self, idx=0):
        self.plot_slice(idx)
        img_alpha = self.create_img_alpha(self.pred_slice.T)
        self.ax.clear()

        # Define the plot extents for the images to place them side by side
        seismic_extent = [0, self.seis_slice.shape[1], 0, self.seis_slice.shape[0]]
        prediction_extent = [self.seis_slice.shape[1], 2 * self.seis_slice.shape[1], 0, self.pred_slice.shape[0]]

        # Display seismic data on the left
        self.ax.imshow(self.seis_slice.T, cmap=self.cmap_bg, extent=seismic_extent)
        # Display predictions on the right
        self.ax.imshow(img_alpha, alpha=0.5, extent=prediction_extent)

        # Set limits for x-axis and y-axis to encompass both images
        self.ax.set_xlim([0, 2 * self.seis_slice.shape[1]])
        self.ax.set_ylim([self.seis_slice.shape[0], 0])

        # Set title and labels using the dimension_labels stored in the instance
        self.ax.set_title(f"{self.dimension_labels[self.flag_slice]} Seismic Section {idx} and Lithofacies Classes {idx}", pad=20)
        self.ax.set_xlabel('Spatial Coordinates')
        self.ax.set_ylabel('Depth')
        self.fig.canvas.draw_idle()

    def sliders_on_changed(self, val):
        idx = int(val)
        self.imshow_alpha(idx)

    def plot_slice(self, idx):
        """Selects the appropriate slices based on idx and flag_slice."""
        if self.flag_slice == 0:
            self.seis_slice = self.seis_vlm[idx, :, :]
            self.pred_slice = self.pred_vlm[idx, :, :]
        elif self.flag_slice == 1:
            self.seis_slice = self.seis_vlm[:, idx, :]
            self.pred_slice = self.pred_vlm[:, idx, :]
        elif self.flag_slice == 2:
            self.seis_slice = self.seis_vlm[:, :, idx]
            self.pred_slice = self.pred_vlm[:, :, idx]
            
    def create_img_alpha(self, img_input):
        """Create RGBA image with distinct colors for each class."""
        colors = np.array([
            # [0, 0, 0, 0],    # Background - black, semi-transparent
            [0.5, 0.5, 0.5, 0.5],
            [1, 0, 0, 0.5],  # Class 1 - Red, semi-transparent
            [0, 1, 0, 0.5],  # Class 2 - Green, semi-transparent
            [0, 0, 1, 0.5],  # Class 3 - Blue, semi-transparent
            [1, 1, 0, 0.5],  # Class 4 - Yellow, semi-transparent
            [0, 1, 1, 0.5],  # Class 5 - Cyan, semi-transparent
            [1, 0, 1, 0.5]   # Class 6 - Magenta, semi-transparent
        ])
        # Allocate RGBA image based on input shape and apply colors based on class index
        img_alpha = np.zeros((*img_input.shape, 4))
        for i in range(1, 7):  # Assuming class indices start from 1
            img_alpha[img_input == i] = colors[i]
        return img_alpha

    def reset_button_on_clicked(self, event):
        """Resets the slider to the initial value when the reset button is clicked."""
        self.idx_slider.reset()
