# coding = utf-8
import numpy as np
import os
import pandas as pd
import random


def sliding_window(data, window, overlap=0):
    """ 
    Segment a sequence data by a sliding window.

    :param data: a sequence. unit:sample point
    :param window: the length of th window.unit: sample point
    :param overlap: the overlap between the jointly window, default 0.
    :return: frames, shape is [,128,11],column_name=['timestamp','ankle_x','ankle_y','ankle_z','thigh_x','thigh_y','thigh_z','trunk_x','trunk_y','trunk_z','annotation']
    """
    window = int(window)
    overlap = int(overlap)
    length = len(data)
    start = 0
    end = start + window - 1
    items = []
    while end < length:
        if end >= length:
            break
        temp = np.array(data[start:end, :])
        if check_sequence(np.array(temp), 1000 / 64):
            items.append(temp)
        start += window - overlap
        end = start + window - 1
    return items


def make_data_for_rnn(frame):
    """
    put a frame to a 2-dimensional matrix to apply rnn

    :param frame: 
    :return: 
    """


def check_sequence(data, mtime):
    """ 
    Check if the data is a sequence by its timestamp data[:,0] after abandoning some samples whose label is 0.

    :param data:  a sequence data
    :param mtime: the sampling frequency,hz
    :return: Boolean, True or False
    """
    interval = data[1:, 0] - data[0:-1, 0]
    for item in interval:
        if item > mtime * 2:
            # print('Non-sequence occur:', item)
            return False
    return True


def get_feature(frame, col_range):
    """
    Extract features from data frame

    :param frame: the data frame from the function sliding_window and the shape is [128,11], first column is timestamp and the last one is annotation.
    :param col_range: the columns that features come from
    :return: [timestamp, features, label]
    """

    timestamp = frame[-1, 0]
    label = frame[-1, -1]
    features = [timestamp]
    for i in col_range:
        c_data = np.array(frame[:, i])
        c_data = c_data * 0.001
        f_mean = np.mean(c_data)
        d_mean = c_data - np.mean(c_data)
        mx = np.max(c_data)
        mn = np.min(c_data)
        if mx == mn:
            return -1
        d_fft = np.fft.fft(d_mean, 128)
        d_1 = abs(d_fft * np.conj(d_fft)) / 128
        # d_mode = np.abs(d_fft) / 256
        # power low
        # PL = x_numericalIntegration(d_1[1:13], 64)
        PL = np.sum(d_1[1:6])
        # Dominant Frequency
        DF = np.argmax(d_1[1:33])
        # Dominant Frequency Amplitude
        DFA = np.max(d_1[1:33])
        # PF = x_numericalIntegration(d_1[13:33], 64)
        PF = np.sum(d_1[7:16])
        TP = PL + PF
        FI = PF / PL
        sd = np.std(c_data)
        var = np.var(c_data)
        t_feature = [f_mean, sd, var, mx, mn, PL, TP, FI]
        features.extend(t_feature)
    features.append(label)
    return features


def get_all_data_from_dir(file_dir):
    """
    Read all the data files in the given direction.

    :param file_dir: The data files' direction, for example './ok_data'
    :return: A 2-dimensional data list
    """
    m_list = os.listdir(file_dir)
    all_data = []
    for files in m_list:
        data = pd.read_csv(os.path.join(file_dir, files))
        all_data.append(data)
    return all_data


def read_data_from(file_path):
    """
    Read the data from the given path.

    :param file_path: The data file's path
    :reture: A data list
    """

    return np.array(pd.read_csv(file_path))


def deal_with_0():
    """
    Delete the annotation 0 and saved into the direction "ok_data" and keep the original file name.
    """
    file_dir = './dataset_fog_release/dataset/'
    file_list = os.listdir(file_dir)
    for file in file_list:
        data = pd.read_csv(os.path.join(file_dir, file), delimiter=' ', header=None)
        data.columns = ['timestamp', 'ankle_hori_fw', 'ankle_vertical', 'ankle_hori_lat', 'thigh_hori_fw',
                        'thigh_vertical',
                        'thigh_hori_lat', 'trunk_hori_fw', 'trunk_vertical', 'trunk_hori_lat', 'annotation']
        temp = data[data["annotation"] != 0]
        pd.DataFrame(temp).to_csv('./ok_data/' + file, index=False)
        print('deal_with_0: {}, data length:{}'.format(file, temp.shape))


def x_numericalIntegration(x, sr):
    """
    computing the summed power of frequency bins.

    :param x: the frequency bins
    :param sr: the sampling frequency
    :return: the summed power
    """
    return (np.sum(x[0:-1]) + np.sum(x[1:])) / (2 * sr)


def print_result(cm):
    """
    Print the specificity and sensitivity based on the confuse matrix.

    :param cm: confuse matrix
    :return: a dict containing 'specificity' and 'sensitivity'
    """
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[1, 0]
    FN = cm[0, 1]
    print("Specifity={}\tSensitivity={}".format(float(TN) / (TN + FP), float(TP) / (TP + FN)))
    return {"specificity": float(TN) / (TN + FP), 'sensitivity': float(TP) / (TP + FN)}


def float_range(start, stop, steps):
    """ 
    Computes a range of floating value.

    :param start:  Start value.
    :param stop: End value.
    :param steps: Number of values
    :return: A list of floats
    """
    return [start + float(i) * (stop - start) / (float(steps) - 1) for i in range(steps)]


def find_fog(Y):
    """
    Find the indices of fogs.

    :param Y: the annotation list
    :return: A dict contain start points and end points
    """
    start = []
    end = []
    delta = Y[1:] - Y[:-1]
    for i in range(len(delta)):
        if delta[i] > 0:
            start.append(i)
        else:
            if delta[i] < 0:
                end.append(i)
    start = np.array(start)
    end = np.array(end)
    start = start + 1
    return {"start": start, "end": end}


def down_sampling(Y, n):
    """
    Down sampling the data to keep the classes balance.

    :param Y: the imbalance annotation list
    :param n: the number that down to
    :return: balanced indices list
    """
    down_set = np.array(random.sample(Y, n))
    return down_set


def abandon_pre_fog(Y, pre_length):
    """
    Abandon the data during the length-time before FOG, to make the normal locomotion's data more pure.

    :param Y: the label list
    :param pre_length: the length of pre-fog(unit: frame)
    :return: the indices of fog, pre_fog and normal
    """
    fogs = find_fog(Y)
    fog_starts = fogs['start']
    fog_ends = fogs['end']
    fog_indices = []
    pre_fog_indices = []
    if fog_starts[0] - pre_length > 0:
        normal_indices = range(0, fog_starts[0] - pre_length)
    else:
        normal_indices = []
    for i in range(len(fog_starts)):
        fog_indices.extend(range(fog_starts[i], fog_ends[i] + 1))
        pre_fog_indices.extend(range(fog_starts[i] - pre_length, fog_starts[i]))
        if i < len(fog_starts) - 1 and fog_ends[i] + pre_length < fog_starts[i + 1] - pre_length:
            normal_indices.extend(range(fog_ends[i] + pre_length, fog_starts[i + 1] - pre_length))
    if fog_ends[-1] + pre_length < len(Y):
        normal_indices.extend(range(fog_ends[-1] + pre_length, len(Y)))
    return {'fog_indices': fog_indices, "pre_fog_indices": pre_fog_indices, "normal_indices": normal_indices}


def get_sequence_data(x, indices, length, time):
    s_data = []
    nums = []
    for i in range(len(indices)):
        if indices[i] > length:
            temp = x[indices[i] - length:indices[i], 1:]
            if check_sequence(x[indices[i] - length:indices[i], :], time):
                s_data.extend(temp)
                nums.append(length)
            else:
                print(x[indices[i] - length:indices[i], 0])
    return {'data': s_data, 'lengths': nums}


def one_hot(y):
    y_one_hot = np.zeros([y.shape[0], 2])
    for i in range(y_one_hot.shape[0]):
        idx = y[i][0]
        y_one_hot[i][idx] = 1
    return y_one_hot


def cal_cm(cm):
    TP = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[1, 1]
    acc = float(TP + TN)/float(TP+FP+FN+TN)
    recall = float(TN)/float(TN+FN)
    return acc, recall
