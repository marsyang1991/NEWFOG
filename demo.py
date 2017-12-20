import pandas as pd
import numpy as np
import os
import preprocess_data as pre
from numpy.lib.stride_tricks import as_strided as ast
import h5py


def split_file(path, filename, filetype='.txt'):
    name = path + filename + filetype
    data = pd.read_csv(name)
    data = pd.DataFrame(data)
    # data = data.as_matrix()
    time = data.iloc[:, 0]
    time_diff = np.diff(time)
    idx = np.where(time_diff > 20)
    idx = idx[0]
    if len(idx) <= 0:
        data.to_csv(filename + str(0) + filetype, header=None, index=None)
    else:
        segment = data[0:idx[0] + 1]
        segment.to_csv(filename + str(0) + filetype, header=None, index=None)
        for i in range(1, len(idx)):
            segment = data[idx[i - 1] + 1:idx[i] + 1]
            segment.to_csv(filename + str(i) + filetype, header=None, index=None)
        segment = data[idx[-1] + 1:]
        segment.to_csv(filename + str(len(idx)) + filetype, header=None, index=None)


def extract_frame_matrix(filename):
    # slide window: length = 1s; overlay = 50%
    raw = pd.read_csv(filename, header=None)
    raw = pd.DataFrame(raw)
    row, col = raw.shape
    start = 0
    end = start + 64
    frame_list = []
    y = []
    data_x, data_y = pre.process_dataset_file(np.array(raw))
    print(data_x.shape, data_y.shape)
    # raw = pd.DataFrame(np.hstack([data_x,data_y]))
    while end <= row:
        sample = raw.iloc[start:end, 1:10]
        sample = sample.as_matrix()
        label = raw.iloc[end - 1, -1]
        y.append(label)
        frame_list.extend(sample)
        start = start + 32
        end = start + 64
    return frame_list, y


def extract_frame_concatenate(filename):
    # slide window: length = 1s; overlay = 50%
    raw = pd.read_csv(filename, header=None)
    raw = pd.DataFrame(raw)
    row, col = raw.shape
    start = 0
    end = start + 64
    frame_list = list()
    y = list()
    data_x, data_y = pre.process_dataset_file(np.array(raw))
    raw = pd.DataFrame(np.hstack([data_x, np.reshape(data_y, [data_y.shape[0], 1])]))
    while end <= row:
        sample = raw.iloc[start:end, 1:19]
        sample = sample.as_matrix()
        sample = sample.reshape((1, 64 * 18))
        label = raw.iloc[end, -1]
        y.append(label)
        frame_list.extend(sample)
        start = start + 32
        end = start + 64
    return frame_list, y


def make_data_for_dnn(one_hot=True):
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data_seq/'
    file_list = os.listdir(dir_path)
    train_set = list()
    train_y = list()
    validate_set = list()
    validate_y = list()
    test_set = list()
    test_y = list()
    for name in file_list:
        name = name.split('.')[0]
        frame_list, my = extract_frame_concatenate(dir_path + name + '.txt')
        if name[:3] in train_s:
            train_set.extend(frame_list)
            train_y.extend(my)
        elif name[:3] in validation_s:
            validate_set.extend(frame_list)
            validate_y.extend(my)
        else:
            test_set.extend(frame_list)
            test_y.extend(my)
    if one_hot:
        train_y = change2one_hot(train_y, 2)
        validate_y = change2one_hot(validate_y, 2)
        test_y = change2one_hot(test_y, 2)
    return {'train_set': train_set, 'train_y': train_y, 'validate_set': validate_set, 'validate_y': validate_y,
            'test_set': test_set, 'test_y': test_y}


def make_data_for_cnn(one_hot=True):
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data_seq/'
    file_list = os.listdir(dir_path)
    train_set = []
    train_y = []
    validate_set = []
    validate_y = []
    test_set = []
    test_y = []
    for name in file_list:
        name = name.split('.')[0]
        frame_list, y = extract_frame_matrix(dir_path + name + '.txt')
        if name[:3] in train_s:
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


def change2one_hot(labels, num_class):
    num = len(labels)
    labels = [i - 1 for i in labels]
    # onehot_encoded = list()
    # for value in labels:
    #     letter = [0 for _ in range(num_class)]
    #     letter[value] = 1
    #     onehot_encoded.append(letter)
    ys = np.unique(labels)
    one_hot = np.zeros([num, num_class], dtype=np.int32)
    for i in range(num):
        one_hot[i, np.argwhere(ys == labels[i])] = 1
    return one_hot


def next_batch(xs, ys, epoch, batch):
    class_one = [i for i, c in enumerate(ys) if c[1] == 1]
    # np.argwhere(ys[:, 1] == 1)
    # class_zero = np.argwhere(ys[:, 0] == 1)
    class_zero = [i for i, c in enumerate(ys) if c[0] == 1]
    # print class_one
    num_one = len(class_one)
    # num_zero = len(class_zero)
    batch_one = batch * num_one / len(ys)
    # print batch_one
    batch_zero = batch - batch_one
    # batch_zero = batch_one
    # print batch_zero
    index_one = class_one[batch_one * epoch:batch_one * (epoch + 1)]
    index_zero = class_zero[batch_zero * epoch:batch_zero * (epoch + 1)]
    index = np.vstack([index_one, index_zero])
    # index = index_one
    batch_xs = []
    batch_ys = []
    for k in range(len(index)):
        batch_xs.append(xs[index[k, 0]])
        batch_ys.append(ys[index[k, 0]])
    # batch_xs = batch_one_x.tolist().extend(batch_zero_x.tolist())
    # batch_ys = batch_one_y.tolist().extend(batch_zero_y.tolist())
    return batch_xs, batch_ys


def extract_frame_matrix2(filename):
    # slide window: length = 1s; overlay = 50%
    # (samples, steps, data_dims)
    raw = pd.read_csv(filename, header=None)
    raw = pd.DataFrame(raw)
    row, col = raw.shape
    start = 0
    end = start + 64
    frame_list = list()
    y = list()
    data_x, data_y = pre.process_dataset_file(np.array(raw))
    raw = pd.DataFrame(np.hstack([data_x, np.reshape(data_y, [data_y.shape[0], 1])]))
    while end <= row:
        sample = raw.iloc[start:end, 1:19]
        sample = sample.as_matrix()
        label = raw.iloc[end - 1, -1]
        y.append(label)
        frame_list.append(sample)
        start = start + 32
        end = start + 64
    return frame_list, y


def extract_frame_matrix_pre_label(filename, look_back):
    # slide window: length = 1s; overlay = 50%
    # (samples, steps, data_dims)
    raw = pd.read_csv(filename)
    raw = pd.DataFrame(raw)
    start = 0
    end = start + 64
    frame_list = list()
    y = list()
    data_x, data_y = pre.process_dataset_file(np.array(raw))
    raw = np.hstack([data_x, np.reshape(data_y, [data_y.shape[0], 1])])
    row, _ = raw.shape
    while end < row:
        sample = raw[start:end, 1:19]
        label = raw[end - 1, -1]
        y.append(label)
        frame_list.append(sample)
        start = start + 32
        end = start + 64
    frame_list = np.array(frame_list)
    y = np.array(y)
    if len(y) <= look_back:
        assert 'wrong timestep'
    frame_list = frame_list[0:-look_back, :, :]
    yy = np.zeros([frame_list.shape[0]])
    for i in range(frame_list.shape[0]):
        yy[i] = 1 if 1 in y[i+1:i+look_back+1] else 0
    return frame_list, yy


def make_data_for_cnn2(one_hot=True):
    # (samples_n, channels_n, dim_input)
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data_seq/'
    file_list = os.listdir(dir_path)
    train_set = list()
    train_y = list()
    validate_set = list()
    validate_y = list()
    test_set = list()
    test_y = list()
    for name in file_list:
        name = name.split('.')[0]
        frame_list, y = extract_frame_matrix2(dir_path + name + '.txt')
        if name[:3] in train_s:
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


def make_data_with_pre_label(look_back, one_hot=True):
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
        frame_list, y = extract_frame_matrix_pre_label(dir_path + name + '.txt', look_back=look_back)
        if name[:3] in train_s:
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


def make_sequential_data(timestep):
    """
    make data for rnn, each sample is sequence of frames whose length is timestep 
    :param timestep: np.int16 
    the sequential length
    :param one_hot: boolean
    if the label y in a shape of one-hot
    :return: [samples, timesteps, channels, dims]
    """
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data_seq/'
    file_list = os.listdir(dir_path)
    train_set = np.empty([0, timestep, 64, 18], dtype=np.float32)
    train_y = np.empty([0, timestep], dtype=np.int8)
    validate_set = np.empty([0, timestep, 64, 18], dtype=np.float32)
    validate_y = np.empty([0, timestep], dtype=np.int8)
    test_set = np.empty([0, timestep, 64, 18], dtype=np.float32)
    test_y = np.empty([0, timestep], dtype=np.int8)
    for name in file_list:
        print("File Processing:{0}".format(name))
        name = name.split('.')[0]
        frame_list, y = extract_frame_matrix_sequential(dir_path + name + '.txt', timestep=timestep)
        if name[:3] in train_s:
            train_set = np.append(train_set, frame_list, axis=0)
            train_y = np.append(train_y, y, axis=0)
        elif name[:3] in validation_s:
            validate_set = np.append(validate_set, frame_list, axis=0)
            validate_y = np.append(validate_y, y, axis=0)
        else:
            test_set = np.append(test_set, frame_list, axis=0)
            test_y = np.append(test_y, y, axis=0)
        print(train_set.shape)
        print(validate_set.shape)
        print(test_set.shape)
    return {'train_set': train_set, 'train_y': train_y, 'validate_set': validate_set, 'validate_y': validate_y,
            'test_set': test_set, 'test_y': test_y}


def extract_frame_matrix_sequential(filename, timestep):
    # slide window: length = 1s; overlay = 50%
    # (samples, steps, data_dims)
    raw = pd.read_csv(filename)
    raw = pd.DataFrame(raw)
    data_x, data_y = pre.process_dataset_file(np.array(raw))
    raw = np.hstack([data_x, np.reshape(data_y, [data_y.shape[0], 1])])
    row, _ = raw.shape
    least_length = 64 + (timestep - 1) * 32
    samples, lables = spilt_squential(raw, np.arange(0, raw.shape[0] - least_length, 32), timestep)
    # print(np.unique(lables))
    return samples, lables


def spilt_squential(data, sp, step):
    """
    spilt the data into sequential samples
    :param data: ndarray [samples, channels, dims]
    :param sp: ndarray 
    start points
    :param step: the sequence length 
    :return: [len(sp), step, channels, dims]
    """
    new_samples = np.empty(shape=[0, step, 64, data.shape[1] - 1], dtype=np.float32)
    new_labels = np.empty(shape=[0, step], dtype=np.int8)
    try:
        data = np.array(data, dtype=np.float32)
    except Exception:
        assert "data wrong"
    for p in sp:
        assert p + 64 + (step - 1) * 32 <= data.shape[0], 'wrong start position:{0},{1}'.format(p, data.shape[0])
        all_data = np.array(data[p:p + 64 + (step - 1) * 32, :], dtype=np.float32)
        # print(np.unique(all_data[:,-1]))
        # new_shape = np.array([1, step, 64, 19])
        # new_strides = tuple(np.array(all_data.strides) * np.array([32, 19])) + all_data.strides
        # d = ast(all_data, shape=new_shape, strides=new_strides)
        d = np.empty([1, 0, 64, 19])
        for k in range(step):
            start = k * 32
            end = start + 64
            dd = all_data[start:end, :]
            d = np.concatenate([d, dd.reshape([1, 1, 64, 19])], axis=1)
        cur_data = d[:, :, :, :-1]
        cur_label = np.array(d[:, :, -1, -1], dtype=np.int8)
        cur_label = cur_label.reshape([-1, step])
        new_samples = np.concatenate((new_samples, cur_data), axis=0)
        new_labels = np.concatenate((new_labels, cur_label), axis=0)
    return new_samples, new_labels


def sequential_dataset(segments, labels, look_back=1, batch_size=1):
    nb_rows = 64
    nb_columns = 18
    sequences = np.empty([0, look_back, nb_rows, nb_columns])  # [nb_samples, time_step, nb_rows, nb_columns]
    ys = np.empty([0])
    segments = np.array(segments)  # [n, nb_rows, nb_column]
    # print(segments.shape)
    labels = np.array(labels)
    # print(labels.shape)
    for index in range(segments.shape[0]-look_back-1):
        t = segments[index:index + look_back]
        sequences = np.append(sequences, t.reshape([1, t.shape[0], t.shape[1], t.shape[2]]), axis=0)
        l = labels[index + look_back]
        ys = np.append(ys, l.reshape([1]), axis=0)
    return sequences, ys


def make_sequential_data2(look_back):
    """
    make data for rnn, each sample is sequence of frames which are from the whole file 
    :return: [samples, timesteps, channels, dims]
    samples is the number of the files
    timesteps vary between samples
    channels and dims are the rows and columns of the frames
    """
    train_s = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10']
    validation_s = ['S09']
    # test_S = ['S02']
    dir_path = 'data_seq/'
    file_list = os.listdir(dir_path)
    train_set = np.empty([0, look_back, 64, 18])
    train_y = np.empty([0])
    validate_set = np.empty([0, look_back, 64, 18])
    validate_y = np.empty([0])
    test_set = np.empty([0, look_back, 64, 18])
    test_y = np.empty([0])
    for name in file_list:
        print("File Processing:{0}".format(name))
        name = name.split('.')[0]
        segments, labels = extract_frame_matrix2(dir_path + name + '.txt')  # segmentation by sliding window
        xs, ys = sequential_dataset(segments=segments, labels=labels, look_back=look_back)
        if name[:3] in train_s:
            train_set = np.append(train_set, xs, axis=0)
            train_y = np.append(train_y, ys, axis=0)
        elif name[:3] in validation_s:
            validate_set = np.append(validate_set, xs, axis=0)
            validate_y = np.append(validate_y, ys, axis=0)
        else:
            test_set = np.append(test_set, xs, axis=0)
            test_y = np.append(test_y, ys, axis=0)
    return {'train_set': train_set, 'train_y': train_y, 'validate_set': validate_set, 'validate_y': validate_y,
            'test_set': test_set, 'test_y': test_y}


if __name__ == "__main__":
    data = make_sequential_data2(5)
    print(data['train_set'].shape)
    file = h5py.File('TrainSet_sequence_5.h5', 'w')
    file.create_dataset('train_set_x', data=data['train_set'])
    file.create_dataset('train_set_y', data=data['train_y'])
    file.create_dataset('validate_set_x', data=data['validate_set'])
    file.create_dataset('validate_y', data=data['validate_y'])
    file.create_dataset('test_set_x', data=data['test_set'])
    file.create_dataset('test_set_y', data=data['test_y'])
    file.close()
    # file = h5py.File('TrainSet_sequence_5.h5', 'r')
    # train_set_data = file['train_set_x'][:]
    # train_set_y = file['train_set_y'][:]
    # train_set_img_num = file['train_set_img_num'][:]
    # # .........
    # file.close()
