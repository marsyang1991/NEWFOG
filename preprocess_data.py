import numpy as np
from scipy import signal
import pandas as pd

# Hardcoded number of sensor channels employed in the Daphnet
NB_SENSOR_CHANNELS = 9
# Hardcoded number of the files defining the Daphnet data
DAPHNET_DATA_FILES_TRAIN = [
    'data_seq/S01R010.txt',
    'data_seq/S01R011.txt',
    'data_seq/S01R020.txt',
    'data_seq/S03R010.txt',
    'data_seq/S03R011.txt',
    'data_seq/S03R020.txt',
    'data_seq/S03R030.txt',
    'data_seq/S04R010.txt',
    'data_seq/S04R011.txt',
    'data_seq/S05R010.txt',
    'data_seq/S05R011.txt',
    'data_seq/S05R012.txt',
    'data_seq/S05R013.txt',
    'data_seq/S05R020.txt',
    'data_seq/S05R021.txt',
    'data_seq/S06R010.txt',
    'data_seq/S06R011.txt',
    'data_seq/S06R012.txt',
    'data_seq/S06R020.txt',
    'data_seq/S06R021.txt',
    'data_seq/S07R010.txt',
    'data_seq/S07R020.txt',
    'data_seq/S08R010.txt',
    'data_seq/S08R011.txt',
    'data_seq/S08R012.txt',
    'data_seq/S08R013.txt',
    'data_seq/S10R010.txt',
    'data_seq/S10R011.txt'
]
DAPHNET_DATA_FILES_TEST = [
    'data_seq/S02R010.txt',
    'data_seq/S02R020.txt'
]
DAPHNET_DATA_FILES_VALID = [
    'data_seq/S09R010.txt',
    'data_seq/S09R011.txt',
    'data_seq/S09R012.txt',
    'data_seq/S09R013.txt',
    'data_seq/S09R014.txt'
]


def divide_x_y(data):
    """Segments each sample into (time+features) and (labels)
    
    :param data: numpy integer matrix
    :return: numpy integer matrix, numpy integer array
    """
    data_x = data[:, :10]
    data_y = data[:, -1]
    return data_x, data_y


def adjust_idx_labels(data_y):
    """Transform original labels into the range [0, nb_labels-1]
    
    :param data_y: numpy integer array
    :return: numpy integer array (Modified sensor labels)
    """
    data_y -= 1
    return data_y


def process_dataset_file(data):
    data_x, data_y = divide_x_y(data)
    data_y = adjust_idx_labels(data_y)
    data_y = data_y.astype(int)
    _, data_x = split_data_into_time_acc(data_x)
    data_x = filter_acc(data_x)
    data_x = normalize(data_x)
    return data_x, data_y


def split_data_into_time_acc(data):
    time = data[:, 0]
    acc = data[:, 1:]
    return time, acc


def filter_acc(data_x):
    data_x = np.array(data_x,dtype=np.float32)
    sampling_rate = 64.0
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


if __name__== "__main__":
    filestr = "data_seq/S01R010.txt"
    data = pd.read_csv(filestr, header=None)
    data_x, data_y = process_dataset_file(np.array(data))

