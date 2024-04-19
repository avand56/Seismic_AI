import tensorflow as tf

@tf.function
def evaluate_step(imgs, masks):
    predictions = model(imgs, training=False)  # Set training to False to ensure all layers are in inference mode
    # Compute loss
    loss = loss_fn(masks, predictions)
    # Update each metric
    pixel_accuracy.update_state(masks, predictions)
    mean_cls_acc.update_state(masks, predictions)
    m_iou.update_state(masks, predictions)
    return loss


def evaluate_model(test_dataset):
    # Reset metrics at the start of the evaluation
    pixel_accuracy.reset_states()
    mean_cls_acc.reset_states()
    m_iou.reset_states()

    total_loss = 0
    num_batches = 0

    for imgs, masks in test_dataset:
        loss = evaluate_step(imgs, masks)
        total_loss += loss
        num_batches += 1

    # Calculate the average loss over all batches
    average_loss = total_loss / num_batches
    print("Evaluation results:")
    print(f"Average Loss: {average_loss.numpy()}")
    print(f"Pixel Accuracy: {pixel_accuracy.result().numpy()}")
    print(f"Mean Class Accuracy: {mean_cls_acc.result().numpy()}")
    print(f"Mean IOU: {m_iou.result().numpy()}")

    # Optionally, return metrics if you need to use them in further analysis
    return {
        "loss": average_loss.numpy(),
        "pixel_accuracy": pixel_accuracy.result().numpy(),
        "mean_class_accuracy": mean_cls_acc.result().numpy(),
        "mean_iou": m_iou.result().numpy()
    }
