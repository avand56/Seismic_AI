import numpy as np
from models.SOM import SOMTensorFlow

# Generate some random data
data = np.random.random((100, 3)).astype(np.float32)

# Initialize and train the SOM
som_tf = SOMTensorFlow(10, 10, 3)
for epoch in range(100):  # Number of training epochs
    for sample in data:
        som_tf.train(data, 1)  # Train with one iteration for online learning

# After training use som_tf.weights to analyze the trained SOM.
