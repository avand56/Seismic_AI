@tf.function
def train_step(imgs, masks):
    with tf.GradientTape() as tape:
        predictions = model(imgs, training=True)
        loss = loss_fn(masks, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update metrics
    pixel_accuracy(masks, predictions)
    class_acc(masks, predictions)
    mean_cls_acc(masks, predictions)
    m_iou(masks, predictions)

    return loss

def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"\nStart of Epoch {epoch+1}")
        for step, (imgs, masks) in enumerate(dataset):
            loss = train_step(imgs, masks)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss))
                )
                print(f"Seen so far: {(step + 1) * 32} samples")

        # Display metrics at the end of each epoch.
        train_acc = pixel_accuracy.result()
        class_accuracy_res = mean_cls_acc.result()
        miou_res = m_iou.result()
        print(f"Training acc over epoch: {train_acc}")
        print(f"Mean class accuracy: {class_accuracy_res}")
        print(f"Mean IOU: {miou_res}")

        # Reset training metrics at the end of each epoch
        pixel_accuracy.reset_states()
        class_acc.reset_states()
        mean_cls_acc.reset_states()
        m_iou.reset_states()
