import numpy as np
from keras.models import Model, load_model
from keras.layers import *
from keras.layers.pooling import MaxPool1D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard, EarlyStopping
from demo import *
from Config import config_rnn
import Util


def load_data(one_hot=True):
    file_list = ['S01R010.txt', 'S01R011.txt', 'S01R020.txt', 'S02R010.txt', 'S02R020.txt', 'S03R010.txt',
                 'S03R011.txt',
                 'S03R020.txt', 'S03R030.txt', 'S04R010.txt', 'S04R011.txt', 'S05R010.txt', 'S05R011.txt',
                 'S05R012.txt',
                 'S05R013.txt', 'S05R020.txt', 'S05R021.txt', 'S06R010.txt', 'S06R011.txt', 'S06R012.txt',
                 'S06R020.txt',
                 'S06R021.txt', 'S07R010.txt', 'S07R020.txt', 'S08R010.txt', 'S08R011.txt', 'S08R012.txt',
                 'S08R013.txt',
                 'S09R010.txt', 'S09R011.txt', 'S09R012.txt', 'S09R013.txt',
                 'S09R014.txt', 'S10R010.txt', 'S10R011.txt']
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data/'
    # file_list = os.listdir(dir_path)
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


def build_model(config):
    input1 = Input(shape=config.input_shape, name='input')
    conv1 = TimeDistributed(
        Conv1D(filters=config.conv_filters, kernel_size=config.conv_kernal_size, activation='relu'), name='conv')(
        input1)
    pooling1 = TimeDistributed(MaxPool1D(), name='pooling')(conv1)
    flat = TimeDistributed(Flatten(), name='flat')(pooling1)
    lstm_encoder = LSTM(units=config.lstm_units, return_sequences=True, name='lstm_encoder')(flat)
    # dp1 = Dropout(rate=config.dropout_rate, name="dropout")(lstm1)
    a = attention(lstm_encoder)
    # a = Flatten()(a)
    lstm_decoder = LSTM(units=64)(a)
    fc = Dense(units=config.fc_units, activation='relu', name='fc')(lstm_decoder)
    output = Dense(units=config.output_units, activation='softmax', name='output')(fc)
    model = Model(inputs=input1, outputs=output, name='rnn')
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['acc'])
    print(model.summary())
    return model


def attention(inputs):
    # [batch_size, time_step, input_dim]
    # a = Dense(5, activation='softmax')(a)
    a = Dense(1)(inputs)
    a = Activation(activation='softmax')(a)
    output_attention = Multiply()([a, inputs])
    # output_attention = merge([a_probs, inputs], mode='mul', name='attention_mul')
    return output_attention


if __name__ == "__main__":
    build_model(config_rnn())
    train_set, train_set_y, validate_set, validate_set_y, test_xs, test_ys = load_data()
    new_train_set, new_train_set_y = Util.balance_training_data(train_set, train_set_y)
    # # print(
    # #     "train_set is {0},validate_set is {1},test_set is {2}".format(train_set.shape, validate_set.shape,
    # #                                                                   test_set.shape))
    training = True
    config = config_rnn()
    if True is training:
        model = build_model(config)
        model.fit(x=new_train_set, y=new_train_set_y, batch_size=config.train_batch_size, epochs=config.train_epoch,
                  validation_data=(validate_set, validate_set_y),
                  callbacks=[TensorBoard(log_dir='./log/rnn_attention'), EarlyStopping(verbose=1)])
        model.save('rnn_attention')
    else:
        model = load_model('rnn_attention')
    y_pred = np.argmax(model.predict(test_xs), axis=1)
    y_true = np.argmax(test_ys, 1)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    print(cm)
    # print(accuracy_score(y_true=y_true, y_pred=y_pred))
    content = np.column_stack([np.array(model.predict(test_xs)), y_pred, y_true])
    result1 = pd.DataFrame(content, columns=['normal', 'FoG', 'y_pred', 'y_true'])
    result1.to_csv('./attention_result.csv')
