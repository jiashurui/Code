import glob
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils.slidewindow import slide_window2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler(feature_range=(-1, 1))


def get_change_points_excluding_first(file_path):
    import pandas as pd
    import numpy as np

    # 读取文件
    data = pd.read_csv(file_path, header=None)

    # 找到行号从哪些位置开始数字变化
    change_points = np.where(data[0].ne(data[0].shift()))[0]

    return change_points.tolist()

def get_uci_data():
    train_data, train_label = get_type_data('train')
    test_data, test_label = get_type_data('test')

    return train_data, train_label, test_data, test_label

def get_type_data(data_type):
    # 加速度
    file_acc_x = f'../data/UCI/{data_type}/Inertial Signals/total_acc_x_{data_type}.txt'
    file_acc_y = f'../data/UCI/{data_type}/Inertial Signals/total_acc_y_{data_type}.txt'
    file_acc_z = f'../data/UCI/{data_type}/Inertial Signals/total_acc_z_{data_type}.txt'

    # 角速度
    file_gyro_x = f'../data/UCI/{data_type}/Inertial Signals/body_gyro_x_{data_type}.txt'
    file_gyro_y = f'../data/UCI/{data_type}/Inertial Signals/body_gyro_y_{data_type}.txt'
    file_gyro_z = f'../data/UCI/{data_type}/Inertial Signals/body_gyro_z_{data_type}.txt'
    label_file = f'../data/UCI/{data_type}/y_{data_type}.txt'

    # 读取数据
    data_acc_x = pd.read_csv(file_acc_x, sep='\\s+', header=None)
    data_acc_y = pd.read_csv(file_acc_y, sep='\\s+', header=None)
    data_acc_z = pd.read_csv(file_acc_z, sep='\\s+', header=None)
    data_gyro_x = pd.read_csv(file_gyro_x, sep='\\s+', header=None)
    data_gyro_y = pd.read_csv(file_gyro_y, sep='\\s+', header=None)
    data_gyro_z = pd.read_csv(file_gyro_z, sep='\\s+', header=None)

    # 标签
    data_l = pd.read_csv(label_file, sep='\\s+', header=None)
    labels = data_l.to_numpy().ravel()
    data_combined = np.stack((
        data_acc_x.to_numpy(),
        data_acc_y.to_numpy(),
        data_acc_z.to_numpy(),
        data_gyro_x.to_numpy(),
        data_gyro_y.to_numpy(),
        data_gyro_z.to_numpy()
    ), axis=-1)

    return data_combined, labels


# TODO 这个代码有问题,标签取成了人
def get_data_1d_uci():
    # 创建示例输入数据 TODO
    file_x = '../data/UCI/train/Inertial Signals/total_acc_x_train.txt'
    file_y = '../data/UCI/train/Inertial Signals/total_acc_y_train.txt'
    file_z = '../data/UCI/train/Inertial Signals/total_acc_z_train.txt'
    file_subject = '../data/UCI/train/subject_train.txt'

    data_x = pd.read_csv(file_x, sep='\\s+', header=None)
    data_y = pd.read_csv(file_y, sep='\\s+', header=None)
    data_z = pd.read_csv(file_z, sep='\\s+', header=None)

    index_of_change = get_change_points_excluding_first(file_subject)

    final_list = []

    for i in (range(len(index_of_change) - 1)):
        start = index_of_change[i]
        end = index_of_change[i + 1]

        index_x_subject = data_x[start:end]
        index_y_subject = data_y[start:end]
        index_z_subject = data_z[start:end]

        flattened_x_array = index_x_subject.values.flatten()
        flattened_y_array = index_y_subject.values.flatten()
        flattened_z_array = index_z_subject.values.flatten()

        combined_array = np.column_stack((flattened_x_array, flattened_y_array, flattened_z_array))
        data_sliced_list = slide_window2(combined_array, 200, 0.5)

        final_list.extend(data_sliced_list)

    # shuffle data
    random.shuffle(final_list)

    test_data = torch.tensor(np.array(final_list), dtype=torch.float32).transpose(1, 2)
    labels = torch.zeros(test_data.shape[0], dtype=torch.long)
    return test_data, labels

# 读取的是UCI public的数据
# 这个读取的是561维度的特征
# 并非原始的加速度,角速度数据
def get_data_1d_uci_all_features(slide_window_length):
    train_data_file = '../data/UCI/train/X_train.txt'
    train_label_file = '../data/UCI/train/y_train.txt'

    test_data_file = '../data/UCI/test/X_test.txt'
    test_label_file = '../data/UCI/test/y_test.txt'

    train_data = pd.read_csv(train_data_file, sep='\\s+', header=None)
    train_label = pd.read_csv(train_label_file, sep='\\s+', header=None)
    test_data = pd.read_csv(test_data_file, sep='\\s+', header=None)
    test_label = pd.read_csv(test_label_file, sep='\\s+', header=None)


    train_data_tensor = np.array(train_data)
    train_label_tensor = np.array(train_label)
    test_data_tensor = np.array(test_data)
    test_label_tensor = np.array(test_label)

    return train_data_tensor, train_label_tensor, test_data_tensor, test_label_tensor


# 按照需求,返回训练数据(指定某一种标签)
# 用于做异常检测
# @Param: data_type:{'train'|'test'}
def get_data_1d_uci_part_data(data_type):
    file_acc_x = f'../data/UCI/{data_type}/Inertial Signals/total_acc_x_{data_type}.txt'
    file_acc_y = f'../data/UCI/{data_type}/Inertial Signals/total_acc_y_{data_type}.txt'
    file_acc_z = f'../data/UCI/{data_type}/Inertial Signals/total_acc_z_{data_type}.txt'

    file_gyro_x = f'../data/UCI/{data_type}/Inertial Signals/body_gyro_x_{data_type}.txt'
    file_gyro_y = f'../data/UCI/{data_type}/Inertial Signals/body_gyro_y_{data_type}.txt'
    file_gyro_z = f'../data/UCI/{data_type}/Inertial Signals/body_gyro_z_{data_type}.txt'

    label_file = f'../data/UCI/{data_type}/y_{data_type}.txt'
    data_l = pd.read_csv(label_file, sep='\\s+', header=None)

    data_acc_x = pd.read_csv(file_acc_x, sep='\\s+', header=None)
    data_acc_y = pd.read_csv(file_acc_y, sep='\\s+', header=None)
    data_acc_z = pd.read_csv(file_acc_z, sep='\\s+', header=None)
    data_gyro_x = pd.read_csv(file_gyro_x, sep='\\s+', header=None)
    data_gyro_y = pd.read_csv(file_gyro_y, sep='\\s+', header=None)
    data_gyro_z = pd.read_csv(file_gyro_z, sep='\\s+', header=None)
    labels = data_l.to_numpy().ravel()
    data_combined = np.stack((
        data_acc_x.to_numpy(),
        data_acc_y.to_numpy(),
        data_acc_z.to_numpy(),
        data_gyro_x.to_numpy(),
        data_gyro_y.to_numpy(),
        data_gyro_z.to_numpy()
    ), axis=-1)

    # 1 WALKING 5 STANDING
    selected_data = data_combined[(labels != 2)]
    not_selected_data = data_combined[(labels == 2)]

    # 归一化处理，将值缩放到 [-1, 1] 范围 TODO 用在autoencoder上面发现重建成了一个直线
    # selected_data = 2 * (selected_data - selected_data.min()) / (selected_data.max() - selected_data.min()) - 1
    # not_selected_data = 2 * (not_selected_data - not_selected_data.min()) / (not_selected_data.max() - not_selected_data.min()) - 1


    train_data_tensor = torch.tensor(np.array(selected_data), dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(np.array(not_selected_data), dtype=torch.float32).to(device)

    return train_data_tensor, test_data_tensor


# 返回UCI数据集所有数据
def get_data_1d_uci_all_data():
    train_data_normal, train_data_abnormal = get_data_1d_uci_part_data('train')
    test_data_normal, test_data_abnormal = get_data_1d_uci_part_data('test')

    return train_data_normal, train_data_abnormal, test_data_normal, test_data_abnormal

# 返回UCI数据集所有数据(不分train set和 test set)
# 只返回
def get_uci_all_data():
    train_data_normal, train_data_abnormal = get_data_1d_uci_part_data('train')
    test_data_normal, test_data_abnormal = get_data_1d_uci_part_data('test')


    normal = torch.cat((train_data_normal, test_data_normal), dim=0)
    abnormal = torch.cat((train_data_abnormal, test_data_abnormal), dim=0)
    return normal , abnormal

if __name__ == '__main__':
    t, tt = get_uci_all_data()
    print()
