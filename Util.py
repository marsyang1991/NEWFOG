import os
import pandas as pd
import numpy as np
from scipy import signal
import random
import keras.backend as K


class FOG:
    def __init__(self):
        self.fogs = self.find_FOG()

    def find_FOG(self):
        """
        find all the FOG's position
        :return: [('filename', start, end)]
        """
        dir_path = 'data/'
        file_list = os.listdir(dir_path)
        fogs = list()
        for file_name in file_list:
            ff = []
            data = pd.read_csv(dir_path + file_name, header=None)
            data = np.array(data)
            edge = data[1:, -1] - data[:-1, -1]
            starts = np.argwhere(edge == 1)
            ends = np.argwhere(edge == -1)
            assert starts.shape[0] == ends.shape[0]
            # starts, ends = self.__unite(starts=starts, ends=ends, threshold=64)
            for i in range(starts.shape[0]):
                assert starts[i] < ends[i], "start is no larger than end"
                fog = (starts[i] + 1, ends[i] + 1)
                ff.append(fog)
            fogs.append((file_name, ff))
        return fogs

    def count(self):
        return len(self.fogs)

    def get_length(self):
        lengths = np.empty([self.count()])
        for i in range(self.count()):
            lengths[i] = self.fogs[i][2] - self.fogs[i][1]
        return lengths

    def __unite(self, starts, ends, threshold=64):
        starts = np.array(starts)
        ends = np.array(ends)
        i = 0
        while i < ends.shape[0] - 1:
            if starts[i + 1] - ends[i] <= threshold:
                starts = np.delete(starts, i + 1)
                ends = np.delete(ends, i)
                i -= 1
            i += 1
        return starts, ends

    def get_pre_fog(self, time=10):
        pre_fog = list()
        sampling_rate = 64
        for i in self.fogs:
            filename = i[0]
            e = np.array(i[1])
            data = pd.read_csv('data/' + filename, header=None)
            data = np.array(data)
            for i in range(e.shape[0]):
                if e[i, 0] < time * sampling_rate:
                    s = 0
                else:
                    s = e[i, 0] - time * sampling_rate
                pre = data[int(s):int(e[i, 0]), :]
                if 2 in pre[:, -1]:
                    print filename
                pre_fog.append(pre)
        pre_fog = np.array(pre_fog)
        print(pre_fog.shape[0])
        return pre_fog


def annotate_pre_post(data_seqs, pre_time=2 * 64):
    # data_seqs : ndarray, [timestamps, data_dims]
    # fogs: ndarray, [start,end]
    # pre_time: int, the number of samples before FoG
    # post_time: int, the number of samples after FoG
    # return: ndarray, [timestamps, data_dims]
    #         new_data
    new_data = data_seqs
    fogs = np.array(get_fogs(data_seqs))
    for j in range(fogs.shape[0]):
        pre_start = fogs[j, 0] - pre_time
        if pre_start < 0:  # ensure the index larger than 0
            pre_start = 0
        else:
            # the index is larger than former FoG's end point
            if j > 0 and pre_start < fogs[j - 1, 1]:
                pre_start = fogs[j - 1, 1]
        new_data[int(pre_start):int(fogs[j, 0]), -1] = 3
    return new_data


def get_raw(filename):
    data = np.array(pd.read_csv(filename, header=None))
    return data[:, 1:]


def get_fogs(raw):
    """
    new data with annotation "pre_fog"(as 3)
    :return: same shape with get_raw()
    """
    ys = raw[:, -1]
    edge = ys[1:] - ys[:-1]
    starts = np.argwhere(edge == 1)
    ends = np.argwhere(edge == -1)
    ff = list()
    assert starts.shape[0] == ends.shape[0]
    for ii in range(starts.shape[0]):
        assert starts[ii] < ends[ii], "start is no larger than end"
        fog = [starts[ii] + 1, ends[ii] + 1]
        ff.append(fog)
    return ff


def extract_frame_matrix(raw, window, step, sampling_rate):
    # slide window: length = 1s; overlay = 50%
    # (samples, steps, data_dims)
    row, col = raw.shape
    start = 0
    end = start + window
    frame_list = list()
    y = list()
    data_x, data_y = process_dataset_file(raw, sampling_rate=sampling_rate)
    while end <= row:
        sample = data_x[start:end, :]
        label = data_y[end - 1]
        y.append(label)
        frame_list.append(sample)
        start = start + step
        end = start + window
    y = np.array(y, dtype=np.int8)
    return frame_list, list(y)


def divide_x_y(data):
    """Segments each sample into (time+features) and (labels)

    :param data: numpy integer matrix
    :return: numpy integer matrix, numpy integer array
    """
    data_x = data[:, :-1]
    data_y = data[:, -1]
    return data_x, data_y


def adjust_idx_labels(data_y):
    """Transform original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
    :return: numpy integer array (Modified sensor labels)
    """
    data_y -= 1
    return data_y


def process_dataset_file(data, sampling_rate):
    data_x, data_y = divide_x_y(data)
    data_y = adjust_idx_labels(data_y)  # make the label [0,1]
    data_y = data_y.astype(int)
    data_x = filter_acc(data_x, sampling_rate=sampling_rate)
    data_x = normalize(data_x)
    return data_x, data_y


def filter_acc(data_x, sampling_rate):
    data_x = np.array(data_x, dtype=np.float32)
    # sampling_rate = 64.0
    nyq_freq = sampling_rate / 2.0
    new_channels = []
    for channel in data_x.transpose():
        gravity = butter_lowpass_filter(channel, 0.3, nyq_freq=nyq_freq, order=3)
        body = channel
        body -= gravity
        new_channels.append(body)
        new_channels.append(gravity)
    preproceed_data = np.array(new_channels).transpose()
    return preproceed_data


def normalize(x):
    """
    Normalize all sensor channels by mean substraction
    dividing by the standard deviation and by 2
    :param x: numpy integer matrix
        sensor data
    :return: 
        Normalised sensor data
    """
    x = np.array(x, dtype=np.float32)
    x /= 1000.0
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 1e-7
    x /= (std * 2)
    return x


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def make_sequence(xs, ys, length):
    '''
    make sequence for RNN
    :param xs: [n_frame, n_rows=64, n_feature_dims]
    :param ys: [n_frame]
    :param length: int
    :return: new_xs, new_ys 
    new_xs: [n_frame-length+1, length, n_rows, n_feature_dims]
    new_ys: [n_frame-length+1]
    '''
    xs = np.array(xs)
    ys = np.array(ys)
    [n_frame, n_rows, n_dims] = xs.shape
    new_xs = np.empty([n_frame - length, length, n_rows, n_dims])
    new_ys = np.empty([n_frame - length])
    for index in range(length, n_frame):
        new_x = xs[index - length:index, :, :]
        new_y = ys[index - 1]
        new_xs[index - length] = new_x
        new_ys[index - length] = new_y
    return new_xs, new_ys


def balance_training_data(xs, ys):
    yy = np.argmax(ys, axis=1)
    labels = np.unique(yy)
    xs = np.array(xs)
    n_labels = np.empty([len(labels)])
    indice_class = list()
    for i in range(len(labels)):
        ind = np.argwhere(yy == labels[i])
        n_labels[i] = len(ind)
        indice_class.append(ind)
    number = int(np.min(n_labels))
    new_indice = list()
    for i in range(len(indice_class)):
        ran = random.sample(range(0, int(n_labels[i])), number)
        new_indice.extend(indice_class[i][ran])
    new_indice = np.array(new_indice).reshape([-1]).transpose()
    new_xs = xs[new_indice]
    new_ys = ys[new_indice]
    print("new data shape is {0}".format(new_xs.shape))
    return new_xs, new_ys


def delete_near(xs, ys, my_near=10):
    """
    delete samples near y=1
    :param xs: 
    :param ys: 
    :param my_near:
    :return: new_xs, new_ys
    """
    ys = np.array(ys)
    xs = np.array(xs)
    edge = ys[:-1] - ys[1:]
    starts = np.argwhere(edge == 1)
    ends = np.argwhere(edge == -1)
    starts = starts + 1
    ll = np.empty([0])
    for i in starts:
        i = i[0]
        l = np.arange(start=i - my_near, stop=i - 1, step=1)
        ll = np.append(ll, l, axis=0)
    for i in ends:
        l = np.arange(start=i + 1, stop=i + my_near, step=1)
        ll = np.append(ll, l, axis=0)
    ind = 0
    while ind < ll.shape[0]:
        l = int(ll[ind])
        if l < 0 or l > ys.shape[0] or ys[l] == 1:
            ll = np.delete(ll, ind)
            ind -= 1
        ind += 1
    xs = np.delete(xs, ll, axis=0)
    ys = np.delete(ys, ll, axis=0)
    return xs, ys


def recall(y_pred, y_true):
   y_pred = K.argmax(y_pred, axis=1)
