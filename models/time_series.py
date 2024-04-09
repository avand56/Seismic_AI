import tensorflow as tf
from keras import layers, models

class DeepLSTM(models.Model):
    def __init__(self, units, num_layers, input_shape, num_classes):
        """
        Initializes the Deep LSTM model.

        Args:
        - units (int): The number of units in each LSTM layer.
        - num_layers (int): The number of LSTM layers to stack.
        - input_shape (tuple): The shape of the input data (time steps, features).
        - num_classes (int): The number of output classes.
        """
        super(DeepLSTM, self).__init__()
        self.lstm_layers = [layers.LSTM(units, return_sequences=True if i<num_layers-1 else False) for i in range(num_layers)]
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = inputs
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        x = self.dropout(x)
        return self.dense(x)
    

class DeepGRU(models.Model):
    def __init__(self, units, num_layers, input_shape, num_classes):
        """
        Initializes the Deep GRU model.

        Args:
        - units (int): The number of units in each GRU layer.
        - num_layers (int): The number of GRU layers to stack.
        - input_shape (tuple): The shape of the input data (time steps, features).
        - num_classes (int): The number of output classes.
        """
        super(DeepGRU, self).__init__()
        self.gru_layers = [layers.GRU(units, return_sequences=True if i<num_layers-1 else False) for i in range(num_layers)]
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = inputs
        for gru_layer in self.gru_layers:
            x = gru_layer(x)
        x = self.dropout(x)
        return self.dense(x)


lstm_model = DeepLSTM(units=64, num_layers=3, input_shape=(100, 1), num_classes=10)
gru_model = DeepGRU(units=64, num_layers=3, input_shape=(100, 1), num_classes=10)

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow will automatically use the GPU for training.")
else:
    print("TensorFlow could not find a GPU. Training will proceed on a CPU, which may be slower.")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device('/GPU:0'):
    # Compile your model
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_history = lstm_model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=20, batch_size=32)
    gru_history = gru_model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=20, batch_size=32)

    lstm_test_loss, lstm_test_acc = lstm_model.evaluate(test_data, test_labels)
    gru_test_loss, gru_test_acc = gru_model.evaluate(test_data, test_labels)

# 2. Reading the dataset from TFRecords
# raw_dataset = tf.data.TFRecordDataset("your_data.tfrecords")
# parsed_dataset = raw_dataset.map(parse_tfrecord_function)  # Parsing function not shown


