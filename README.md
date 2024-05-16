# Seismic_AI
Library to automate and build seismic interpretation and exploration ML models

The purpose of this model was to showcase capabilities in working with seismic data, including pre-processing seismic data, creating labels and masks, creating predictions, and building re-usable infrastructure that can be used across models we plan to build.

To do this, we created a lithofacies classification U-Net model, which is half of model 4 in our workplan. This model uses F3 Netherlands open source seismic reflection data acquired in the north sea to predict the lithofacies in the data in 2D, providing valuable geology information for our ensemble model. With more data, this model can be readily extended to work in 3D on seismic cubes instead of just 2D patches and can have improved performance.

For more information:
- notebooks/Training.ipynb for training routine
- models/CNNs for the U-Net architecture
- utils for programs to read, write, alter .segy files to .npy for use with model building
- models for other model architecture we plan to implement
- geophysics to compute DHI, conduct basic AVO to create interpretations to be labeled