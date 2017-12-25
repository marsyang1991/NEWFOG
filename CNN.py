from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import *
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
# from demo import *
from utils import *
import Util

input_shape = (64, 18)
conv_filters = 16
kernal_size = 3
full_connect_units = 64
drop_out_rate = 0.5
output_dim = 3
near = 10
pd.DataFrame()


def load_data(one_hot=True):
    # (samples_n, channels_n, dim_input)
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    # train_s = ['S01', 'S03', 'S04', 'S05','S06','']
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
    for full_name in file_list:
        name = full_name.split('.')[0]
        raw = Util.get_raw(dir_path + full_name)
        new_data = Util.annotate_pre_post(raw, pre_time=3 * 64)
        frame_list, y = Util.extract_frame_matrix(new_data, window=64, step=32, sampling_rate=64)
        if name[:3] in train_s:
            print("Load {0} into training set".format(name))
            # frame_list, y = Util.delete_near(frame_list, y, my_near=near)
            train_set.extend(frame_list)
            train_y.extend(y)
        elif name[:3] in validation_s:
            print("Load {0} into validation set".format(name))
            validate_set.extend(frame_list)
            validate_y.extend(y)
        else:
            print("Load {0} into testing set".format(name))
            test_set.extend(frame_list)
            test_y.extend(y)

    if one_hot:
        train_y = to_categorical(train_y, output_dim)
        validate_y = to_categorical(validate_y, output_dim)
        test_y = to_categorical(test_y, output_dim)
    return np.array(train_set), train_y, np.array(validate_set), validate_y, np.array(test_set), test_y


def build_model():
    input1 = Input(shape=input_shape)
    conv1 = Conv1D(conv_filters, kernel_size=kernal_size, padding='same')(input1)
    pooling1 = MaxPool1D()(conv1)
    activation1 = Activation(activation="relu")(pooling1)
    conv2 = Conv1D(conv_filters, kernel_size=kernal_size, padding='same')(activation1)
    pooling2 = MaxPool1D()(conv2)
    activation2 = Activation(activation="relu")(pooling2)
    flatten = Flatten()(activation2)
    full_connect = Dense(full_connect_units, activation="relu")(flatten)
    drop = Dropout(rate=drop_out_rate)(full_connect)
    output = Dense(output_dim, activation="softmax")(drop)
    my_model = Model(inputs=input1, outputs=output, name="cnn")
    return my_model


def precision(y_true, y_pred):
    y_ = K.argmax(y_true, axis=1)
    _y = K.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_pred=_y, y_true=y_)
    return float(cm[1, 1]) / float(cm[0, 1] + cm[1, 1])


def recall(y_true, y_pred):
    y_ = K.argmax(y_true)
    _y = K.argmax(y_pred)
    positive_all = K.sum(y_) + K.floatX
    positive_right = K.sum(y_ * _y)

    # cm = confusion_matrix(y_pred=_y, y_true=y_)
    return positive_right / positive_all


if __name__ == "__main__":
    train_xs, train_ys, validate_xs, validate_ys, test_xs, test_ys = load_data()
    train_xs, train_ys = Util.balance_training_data(train_xs, train_ys)
    training = True
    if True is training:
        model = build_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["acc"])
        print(model.summary())
        model.fit(train_xs, train_ys, epochs=20, batch_size=64, validation_data=[validate_xs, validate_ys],
                  callbacks=[TensorBoard(log_dir='./log/cnn')])
        model.save('model')
    else:
        model = load_model('model')
    y_pred = np.argmax(model.predict(test_xs), axis=1)
    y_true = np.argmax(test_ys, 1)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    print(cm)
    print(accuracy_score(y_true=y_true, y_pred=y_pred))
    content = np.column_stack([np.array(model.predict(test_xs)), y_pred, y_true])
    result1 = pd.DataFrame(content, columns=['normal', 'FoG', 'PreFoG', 'y_pred', 'y_true'])
    result1.to_csv('./result.csv')
