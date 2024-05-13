import numpy as np
import tensorflow as tf

def make_predictions_unet(model, test_data):
    """
    This function takes a trained model and test dataset, performs predictions,
    and returns the predicted lithofacies.

    Args:
    model (tf.keras.Model): The trained segmentation model.
    test_data (tf.data.Dataset): The test dataset prepared in the same format as training data.

    Returns:
    numpy.ndarray: Predicted lithofacies for each test sample.
    """
    predictions = model.predict(test_data)
    # Assuming the model outputs probabilities and need the class with the highest probability
    predicted_classes = np.argmax(predictions, axis=-1)
    return predicted_classes


#########
# To run
#########

# test_dataset = test_dataset.batch(32)  # Adjust batch size as necessary

# # Load your trained model
# model = tf.keras.models.load_model('path_to_your_model.h5')

# # Make predictions
# predicted_lithofacies = make_predictions(model, test_dataset)