class config_rnn:
    def __init__(self):
        self.conv_filters = 16
        self.conv_kernal_size = 3
        self.lstm_units = 128
        self.fc_units = 1024
        self.output_units = 2
        self.input_shape = (5, 64, 18)
        self.dropout_rate = 0.2
        self.train_batch_size = 1
        self.train_epoch = 10

class config_cnn:
    def __init__(self):
        pass

class config_dnn:
    def __init__(self):
        pass