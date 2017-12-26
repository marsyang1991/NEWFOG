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


def load_data(pre=0, one_hot=True):
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data/'
    file_list = os.listdir(dir_path)
    length = 5
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
            new_xs, new_ys = Util.delete_near(new_xs, new_ys, my_near=5)
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


def slice(x, index):
    return x[index, :, :]


def build_model(config):
    input1 = Input(shape=config.input_shape, name='input')
    conv1 = TimeDistributed(
        Conv1D(filters=config.conv_filters, kernel_size=config.conv_kernal_size, activation='relu'), name='conv')(
        input1)
    pooling1 = TimeDistributed(MaxPool1D(), name='pooling1')(conv1)
    flat = TimeDistributed(Flatten(), name='flat1')(pooling1)  # (,time_step, n_row, n_columns)
    lstm1 = LSTM(units=config.lstm_units, return_sequences=False, name='lstm')(flat)
    dp1 = Dropout(rate=config.dropout_rate, name="dropout")(lstm1)
    fc = Dense(units=config.fc_units, activation='relu', name='fc')(dp1)
    output = Dense(units=config.output_units, activation='relu', name='output')(fc)
    cur = Lambda(slice, output_shape=([496]), arguments={'index': -1})(flat)
    output_d = Dense(units=config.output_units, activation='relu', name='output_d')(cur)
    merge1 = Multiply()([output, output_d])

    final_output = Activation(activation='softmax')(merge1)
    my_model = Model(inputs=[input1], outputs=final_output, name='rnn')
    my_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
    print(my_model.summary())
    return my_model


if __name__ == "__main__":
    config = config_rnn()
    # build_model(config)
    train_set, train_set_y, validate_set, validate_set_y, test_set, test_set_y = load_data()
    # new_train_set, new_train_set_y = Util.balance_training_data(train_set, train_set_y)
    training = True
    if True is training:
        model = build_model(config)
        # train_y = to_categorical(train_set_y, num_classes=2)
        # print(y.shape)
        # validate_y = to_categorical(validate_set_y, num_classes=2)
        model.fit(x=train_set, y=train_set_y, batch_size=1, epochs=10,
                  validation_data=([validate_set], validate_set_y),callbacks=[TensorBoard(log_dir='./log/rnn_m/')])
        model.save('rnn')
    else:
        model = load_model('rnn')
    y_ = model.predict(x=[test_set])
    y_pred = np.argmax(y_, axis=1)
    y_true = np.argmax(test_set_y, axis=1)
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    content = np.column_stack([y_, y_pred, y_true])
    result1 = pd.DataFrame(content, columns=['normal', 'FoG', 'y_pred', 'y_true'])
    result1.to_csv('./rnn_result.csv')
    print(matrix)
