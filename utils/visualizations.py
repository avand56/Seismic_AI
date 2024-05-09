from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np


def visualize_seismic(volume):
    """ Visualize a 3D seismic volume or lithofacies classification results. """
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))
    mlab.pipeline.volume(mlab.pipeline.scalar_field(volume), vmin=volume.min(), vmax=volume.max())
    mlab.outline(color=(1, 1, 1))
    mlab.show()

def plot_slices(volume, num_slices=6):
    """ Plot horizontal slices of the 3D volume. """
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    slice_positions = np.linspace(0, volume.shape[2] - 1, num_slices, dtype=np.int)
    for i, ax in enumerate(axes):
        ax.imshow(volume[:, :, slice_positions[i]], cmap='viridis')
        ax.set_title(f'Slice {slice_positions[i]}')
        ax.axis('off')
    plt.show()

def plot_2d_heatmap(data, colormap='viridis'):
    """
    Plots a 2D heatmap using Mayavi, treating the data as a height map.
    
    Args:
    - data (2D numpy array): The 2D array to be visualized as a heatmap.
    - colormap (str): The colormap to use for visualization.
    """
    # Create a figure window
    mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
    
    # Generate grid data
    x, y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    
    # Plotting using surf where 'data' is used as height for 3D visualization
    surf = mlab.surf(x, y, data, colormap=colormap)
    
    # Enhancing the view
    mlab.view(azimuth=0, elevation=90)  # Top view to mimic 2D heatmap
    mlab.colorbar(surf, orientation='vertical', title='Value')
    
    # Show the plot
    mlab.show()