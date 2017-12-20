import numpy as np
from keras.models import Model, load_model
from keras.layers import *
from keras.layers.pooling import MaxPool1D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard, EarlyStopping
from demo import *
from Config import config_rnn
from utils import *


def load_data():
    file = h5py.File('TrainSet_5.h5', 'r')
    train_set = file['train_set_x'][:]
    train_set_y = file['train_set_y'][:]
    validate_set = file['validate_set_x'][:]
    validate_set_y = file['validate_y'][:]
    test_set = file['test_set_x'][:]
    test_set_y = file['test_set_y'][:]
    file.close()
    return train_set, train_set_y, validate_set, validate_set_y, test_set, test_set_y


def build_model(config):
    input1 = Input(shape=config.input_shape, name='input')
    conv1 = TimeDistributed(
        Conv1D(filters=config.conv_filters, kernel_size=config.conv_kernal_size, activation='relu'), name='conv')(
        input1)
    pooling1 = TimeDistributed(MaxPool1D(), name='pooling')(conv1)
    flat = TimeDistributed(Flatten(), name='flat')(pooling1)
    lstm1 = LSTM(units=config.lstm_units, return_sequences=True, name='lstm')(flat)
    # dp1 = Dropout(rate=config.dropout_rate, name="dropout")(lstm1)
    a = attention(lstm1)
    a = Flatten()(a)
    fc = Dense(units=config.fc_units, activation='relu', name='fc')(a)
    output = Dense(units=config.output_units, activation='softmax', name='output')(fc)
    model = Model(inputs=input1, outputs=output, name='rnn')
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
    print(model.summary())
    return model


def my_to_catogrical(y):
    """
    :param y: [None,n_timestep,1]
    :return: [None,n_timestep,2]
    """
    # print(y.shape)
    new = np.empty([y.shape[0], 2])
    for i in range(y.shape[0]):
        new[i] = to_categorical(y[i, -1], num_classes=2)
    return new


def attention(inputs):
    # [batch_size, time_step, input_dim]
    # a = Permute((2, 1))(inputs)
    # a = Dense(5, activation='softmax')(a)
    a = LSTM(5, return_sequences=False)(inputs)
    a = RepeatVector(128)(a)
    a_probs = Permute((2, 1), name="activation_vec")(a)
    output_attention = Multiply()([a_probs, inputs])
    # output_attention = merge([a_probs, inputs], mode='mul', name='attention_mul')
    return output_attention


if __name__ == "__main__":
    # build_model(config_rnn())
    # build_model(config_rnn())
    train_set, train_set_y, validate_set, validate_set_y, test_set, test_set_y = load_data()
    # # print(
    # #     "train_set is {0},validate_set is {1},test_set is {2}".format(train_set.shape, validate_set.shape,
    # #                                                                   test_set.shape))
    training = True
    config = config_rnn()
    if True is training:
        model = build_model(config)
        train_y = to_categorical(train_set_y, num_classes=2)
        # print(y.shape)
        validate_y = to_categorical(validate_set_y, num_classes=2)
        model.fit(x=train_set, y=train_y, batch_size=config.train_batch_size, epochs=config.train_epoch,
                  validation_data=(validate_set, validate_y),
                  callbacks=[TensorBoard(log_dir='./log/rnn'), EarlyStopping(verbose=1)])
        model.save('rnn')
    else:
        model = load_model('rnn')
    y_pred = model.predict(x=test_set)
    y_pred = np.argmax(y_pred.reshape([-1, 2]), axis=1)
    y_true = test_set_y
    print(y_true.shape)
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(matrix)
    print(cal_cm(matrix))
