# Example of adding a debugging step in your data pipeline
def check_shapes(images, labels):
    print("Image shape:", images.shape)  # Should be (batch_size, 128, 128, 1)
    assert images.shape[1:4] == (128, 128, 1), f"Incorrect image shape: {images.shape}"
    return images, labels

def check_dataset_shapes(dataset, expected_image_shape, expected_label_shape):
    """
    Iterate through the dataset and check the shapes of image and label tensors.
    Print information about any tensor that does not match the expected shapes.

    Args:
    - dataset (tf.data.Dataset): The dataset to check.
    - expected_image_shape (tuple): The expected shape of image tensors.
    - expected_label_shape (tuple): The expected shape of label tensors.
    """
    incorrect_shapes = []
    for i, (images, labels) in enumerate(dataset):
        if images.shape[1:] != expected_image_shape or labels.shape[1:] != expected_label_shape:
            incorrect_shapes.append((i, images.shape, labels.shape))
            print(f"Batch {i} has incorrect shapes. Image shape: {images.shape}, Label shape: {labels.shape}")
    
    if not incorrect_shapes:
        print("All tensors in the dataset have the correct shape.")
    else:
        print(f"There were {len(incorrect_shapes)} batches with incorrect shapes.")

    return incorrect_shapes