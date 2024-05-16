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

