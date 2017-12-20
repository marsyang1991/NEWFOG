from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import *
import numpy as np
import pandas as pd
import keras.backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import confusion_matrix
from demo import *
from draw_plot import *
from utils import *

data = make_data_with_pre_label(3, True)
train_xs = np.array(data['train_set'])
train_ys = np.array(data['train_y'])
validate_xs = np.array(data['validate_set'])
validate_ys = np.array(data['validate_y'])
test_xs = np.array(data['test_set'])
test_ys = np.array(data['test_y'])

input_shape = (64, 18)
conv_filters = 16
kernal_size = 3
full_connect_units = 64
drop_out_rate = 0.5
output_dim = 2


def build_model():
    input1 = Input(shape=input_shape)
    conv1 = Conv1D(conv_filters, kernel_size=kernal_size, padding='same')(input1)
    pooling1 = MaxPool1D()(conv1)
    activation1 = Activation(activation="relu")(pooling1)
    flatten = Flatten()(activation1)
    full_connect = Dense(full_connect_units, activation="relu")(flatten)
    drop = Dropout(rate=drop_out_rate)(full_connect)
    output = Dense(output_dim, activation="softmax")(drop)
    model = Model(inputs=input1, outputs=output, name="cnn")
    return model


def recall(y_true, y_pred):
    y_true = K.argmax(y_true, axis=1)
    y_pred = K.argmax(y_pred, axis=1)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(y_true)
    recall = true_positives / possible_positives
    return recall


if __name__ == "__main__":
    training = True
    if True is training:
        model = build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(model.summary())
        model.fit(train_xs, train_ys, epochs=20, batch_size=64, validation_data=[validate_xs,validate_ys],
                  callbacks=[TensorBoard(log_dir='./log/cnn'), EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')])
        model.save('model')
    else:
        model = load_model('model')
    y_pred = np.argmax(model.predict(test_xs), 1)
    y_true = np.argmax(test_ys, 1)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    print(cm)
    print(cal_cm(cm))
    # draw(y_pred, y_true)
