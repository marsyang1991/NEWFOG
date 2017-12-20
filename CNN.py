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
import Util

input_shape = (64, 18)
conv_filters = 16
kernal_size = 3
full_connect_units = 64
drop_out_rate = 0.5
output_dim = 2


def load_data(one_hot=True, near=True):
    # (samples_n, channels_n, dim_input)
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data/'
    file_list = os.listdir(dir_path)
    train_set = list()
    train_y = list()
    validate_set = list()
    validate_y = list()
    test_set = list()
    test_y = list()
    for name in file_list:
        name = name.split('.')[0]
        frame_list, y = Util.extract_frame_matrix(dir_path + name + '.txt')
        if name[:3] in train_s:
            # if near:
            #     frame_list, y = delete_near(frame_list, y)
            train_set.extend(frame_list)
            train_y.extend(y)
        elif name[:3] in validation_s:
            validate_set.extend(frame_list)
            validate_y.extend(y)
        else:
            test_set.extend(frame_list)
            test_y.extend(y)

    if one_hot:
        train_y = change2one_hot(train_y, 2)
        validate_y = change2one_hot(validate_y, 2)
        test_y = change2one_hot(test_y, 2)
    return {'train_set': train_set, 'train_y': train_y, 'validate_set': validate_set, 'validate_y': validate_y,
            'test_set': test_set, 'test_y': test_y}


def delete_near(xs, ys, near=10):
    """
    delete samples near y=1
    :param xs: 
    :param ys: 
    :return: new_xs, new_ys
    """
    ys = np.array(ys)
    xs = np.array(xs)
    edge = ys[:-1] - ys[1:]
    starts = np.argwhere(edge == 1)
    print("starts:{0}".format(starts))
    ends = np.argwhere(edge == -1)
    starts = starts + 1
    ll = np.empty([0])
    for i in starts:
        s = i - near if i <= near else 0
        ll = np.append(ll, np.arange(s, i - 1, 1),axis=0)
    for i in ends:
        s = i + near if i + near < xs.shape[0] else xs.shape[0]
        ll = np.append(ll,np.arange(i, s - 1, 1), axis=0)
    print ll
    xs = np.delete(xs, ll, axis=0)
    ys = np.delete(ys, ll, axis=0)
    return xs, ys


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


if __name__ == "__main__":
    data = load_data()
    train_xs = np.array(data['train_set'])
    train_ys = np.array(data['train_y'])
    validate_xs = np.array(data['validate_set'])
    validate_ys = np.array(data['validate_y'])
    test_xs = np.array(data['test_set'])
    test_ys = np.array(data['test_y'])
    training = True
    if True is training:
        model = build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(model.summary())
        model.fit(train_xs, train_ys, epochs=20, batch_size=64, validation_data=[validate_xs, validate_ys],
                  callbacks=[TensorBoard(log_dir='./log/cnn'),
                             EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')])
        model.save('model')
    else:
        model = load_model('model')
    y_pred = np.argmax(model.predict(test_xs), 1)
    y_true = np.argmax(test_ys, 1)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    print(cm)
    print(cal_cm(cm))
    # draw(y_pred, y_true)
