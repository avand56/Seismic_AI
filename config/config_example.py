import json
import tensorflow as tf

# Load the configuration
with open('config.json') as f:
    config = json.load(f)

# Example: Configuring a model
model_config = config['models'][0]  # For UNet_2D as an example
model = UNet(input_shape=model_config['input_shape'],
             num_classes=model_config['num_classes'],
             dimensionality=model_config['dimensionality'])

model.compile(optimizer=model_config['optimizer'],
              loss=model_config['loss'],
              metrics=model_config['metrics'])

# Proceed to load data, train, and save models as configured
