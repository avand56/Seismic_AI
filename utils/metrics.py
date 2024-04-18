import tensorflow as tf
import numpy as np

class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='confusion_matrix', **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name='cm',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, dtype=tf.int32)

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, weights=sample_weight, dtype=tf.float32)
        self.confusion_matrix.assign_add(cm)

    def result(self):
        return self.confusion_matrix

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

def pixelwise_accuracy(num_classes):
    cm = ConfusionMatrix(num_classes)

    def calc_pixel_accuracy(y_true, y_pred):
        cm.update_state(y_true, y_pred)
        diagonal_sum = tf.linalg.trace(cm.result())
        total_sum = tf.reduce_sum(cm.result())
        return diagonal_sum / total_sum

    return calc_pixel_accuracy

def class_accuracy(num_classes):
    cm = ConfusionMatrix(num_classes)

    def calc_class_accuracy(y_true, y_pred):
        cm.update_state(y_true, y_pred)
        diagonal = tf.linalg.diag_part(cm.result())
        sum_per_class = tf.reduce_sum(cm.result(), axis=1)
        return diagonal / (sum_per_class + tf.keras.backend.epsilon())

    return calc_class_accuracy

def mean_class_accuracy(num_classes):
    class_acc = class_accuracy(num_classes)

    def calc_mean_class_accuracy(y_true, y_pred):
        per_class_acc = class_acc(y_true, y_pred)
        return tf.reduce_mean(per_class_acc)

    return calc_mean_class_accuracy

def class_iou(num_classes):
    cm = ConfusionMatrix(num_classes)

    def calc_class_iou(y_true, y_pred):
        cm.update_state(y_true, y_pred)
        conf_matrix = cm.result()
        true_positive = tf.linalg.diag_part(conf_matrix)
        false_positive = tf.reduce_sum(conf_matrix, axis=0) - true_positive
        false_negative = tf.reduce_sum(conf_matrix, axis=1) - true_positive
        iou = true_positive / (true_positive + false_positive + false_negative + tf.keras.backend.epsilon())
        return iou

    return calc_class_iou

def mean_iou(num_classes):
    class_iou_metric = class_iou(num_classes)

    def calc_mean_iou(y_true, y_pred):
        ious = class_iou_metric(y_true, y_pred)
        return tf.reduce_mean(ious)

    return calc_mean_iou
