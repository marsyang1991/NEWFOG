from keras.models import Model, load_model
from keras.layers import *
from keras.layers.pooling import MaxPool1D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard, EarlyStopping
from keras.engine.topology import Layer
from demo import *
from Config import config_rnn
import os
import Util


def load_data(length=5, one_hot=True):
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data/'
    file_list = os.listdir(dir_path)
    train_set = np.empty([0, length, 64, 18])
    train_y = np.empty([0])
    validate_set = np.empty([0, length, 64, 18])
    validate_y = np.empty([0])
    test_set = np.empty([0, length, 64, 18])
    test_y = np.empty([0])
    for full_name in file_list:
        name = full_name.split('.')[0]
        raw = Util.get_raw(dir_path + full_name)
        # new_data = Util.annotate_pre_post(raw, pre_time=5 * 64)
        frame_list, y = Util.extract_frame_matrix(raw, window=64, step=32, sampling_rate=64)
        new_xs, new_ys = Util.make_sequence(frame_list, y, length=length)
        if name[:3] in train_s:
            print("Load {0} into training set".format(name))
            # new_xs, new_ys = Util.delete_near(new_xs, new_ys, my_near=5)
            train_set = np.append(train_set, new_xs, axis=0)
            train_y = np.append(train_y, new_ys)
        elif name[:3] in validation_s:
            print("Load {0} into validation set".format(name))
            validate_set = np.append(validate_set, new_xs, axis=0)
            validate_y = np.append(validate_y, new_ys)
        else:
            print("Load {0} into testing set".format(name))
            test_set = np.append(test_set, new_xs, axis=0)
            test_y = np.append(test_y, new_ys)
    if one_hot:
        train_y = to_categorical(train_y, 2)
        validate_y = to_categorical(validate_y, 2)
        test_y = to_categorical(test_y, 2)
    return train_set, train_y, validate_set, validate_y, test_set, test_y


def my_loss(y_true, y_pred, output):
    return K.binary_crossentropy(y_true, y_pred) + K.binary_crossentropy(y_pred, output)


def build_model(config, length=5):
    input_shape = [length, 64, 18]
    input1 = Input(shape=input_shape, name='input')
    past = Lambda(lambda x: x[:, :-1, :])(input1)
    conv1 = TimeDistributed(
        Conv1D(filters=config.conv_filters, kernel_size=config.conv_kernal_size, activation='relu'), name='conv')(
        past)
    pooling1 = TimeDistributed(MaxPool1D(), name='pooling1')(conv1)
    flat = TimeDistributed(Flatten(), name='flat1')(pooling1)  # (,time_step, n_row, n_columns)

    lstm1 = LSTM(units=256, return_sequences=False, name='lstm')(flat)
    fc = Dense(units=1024, activation='relu', name='fc')(lstm1)
    dp1 = Dropout(rate=0.5, name="dropout")(fc)
    output = Dense(units=config.output_units, activation='tanh', name='output')(dp1)

    cur = Lambda(lambda x: x[:, -1, :])(input1)
    conv2 = Conv1D(filters=config.conv_filters, kernel_size=config.conv_kernal_size, activation='relu')(cur)
    pool2 = MaxPool1D()(conv2)
    flat2 = Flatten()(pool2)
    d_1 = Dense(units=1024, activation='relu', name="d1")(flat2)
    drop = Dropout(rate=0.5)(d_1)
    output_c = Dense(units=config.output_units, activation='softmax', name='output_d')(drop)


    past = Model(inputs=input1, outputs=output)
    current = Model(inputs=input1, outputs=output_c)
    past.compile(optimizer='adam', loss='mse')
    current.compile(optimizer='adam', loss='binary_crossentropy')

    merge1 = Add()([output, output_c])
    final_output = Activation(activation='softmax')(merge1)

    my_model = Model(inputs=input1, outputs=final_output, name='rnn_modified')
    my_model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['acc'])
    # my_model.compile(optimizer='adam',
    #                  loss=lambda y_true, y_pred: K.binary_crossentropy(y_true, y_pred) + K.binary_crossentropy(
    #                      y_pred, output), metrics=['acc'])
    # print(my_model.summary())
    # print(dense1.summary())
    return my_model, past, current


def set_current_trainable(past, current):
    past.trainable = False
    current.trainable = True


def set_past_trainable(past, current):
    past.trainable = True
    current.trainable = False


if __name__ == "__main__":
    # build_model(config_rnn())
    config = config_rnn()
    length = 5
    # build_model(config)
    train_set, train_set_y, validate_set, validate_set_y, test_set, test_set_y = load_data(length=length)
    train_set, train_set_y = Util.balance_training_data(train_set, train_set_y)
    training = True
    if True is training:
        model, past, current = build_model(config, length=length)
        for epoch in range(10):
            loss = []
            set_past_trainable(past, current)
            for i in range(len(train_set)):
                print(model.train_on_batch(train_set[:i], train_set_y[:i]))
            set_current_trainable(past, current)
            for i in range(len(train_set)):
                loss.append(model.train_on_batch(train_set[:i], train_set_y[:i]))
            print("Epoch #{0}:Loss:{1}".format(epoch, loss[-1]))

        # model.fit(x=train_set, y=train_set_y, batch_size=1, epochs=10,
        #           validation_data=(validate_set, validate_set_y),
        #           callbacks=[TensorBoard(log_dir='./log/rnn_m/'), EarlyStopping(patience=2)])

        model.save('rnn_m')
    else:
        model, dense1, dense2 = load_model('rnn_m')
    y_ = model.predict(x=test_set)
    output_p = dense1.predict(x=test_set)
    output_c = dense2.predict(x=test_set)
    y_pred = np.argmax(y_, axis=1)
    y_true = np.argmax(test_set_y, axis=1)
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    content = np.column_stack([y_, y_pred, y_true, output_p, output_c])
    result1 = pd.DataFrame(content,
                           columns=['normal', 'FoG', 'y_pred', 'y_true', 'ouputP1', 'ouputP2', 'ouputC1', 'ouputC2'])
    result1.to_csv('./rm_result.csv')
    print(matrix)
