import glob
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from utils.slidewindow import slide_window2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_change_points_excluding_first(file_path):
    import pandas as pd
    import numpy as np

    # 读取文件
    data = pd.read_csv(file_path, header=None)

    # 找到行号从哪些位置开始数字变化
    change_points = np.where(data[0].ne(data[0].shift()))[0]

    return change_points.tolist()


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
    file = '../data/UCI/train/X_train.txt'
    label_file = '../data/UCI/train/y_train.txt'

    data_x = pd.read_csv(file, sep='\\s+', header=None)
    data_l = pd.read_csv(label_file, sep='\\s+', header=None)
    data_x['label'] = data_l

    # test value TODO cross validate
    train_values = data_x[(data_x['label'] == 1)]

    filtered_values = data_x[(data_x['label'] != 1)]

    normal_data = slide_window2(train_values, slide_window_length, 0.5)
    abnormal_data = slide_window2(filtered_values, slide_window_length, 0.5)

    train_data_tensor = torch.tensor(np.array(normal_data), dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(np.array(abnormal_data), dtype=torch.float32).to(device)

    return train_data_tensor[:, :, :561], test_data_tensor[:, :, :561]


def get_data_1d_uci_all_data():
    file_acc_x = '../data/UCI/train/Inertial Signals/total_acc_x_train.txt'
    file_acc_y = '../data/UCI/train/Inertial Signals/total_acc_y_train.txt'
    file_acc_z = '../data/UCI/train/Inertial Signals/total_acc_z_train.txt'

    file_gyro_x = '../data/UCI/train/Inertial Signals/body_gyro_x_train.txt'
    file_gyro_y = '../data/UCI/train/Inertial Signals/body_gyro_y_train.txt'
    file_gyro_z = '../data/UCI/train/Inertial Signals/body_gyro_z_train.txt'

    label_file = '../data/UCI/train/y_train.txt'
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

    selected_data = data_combined[labels != 1]
    not_selected_data = data_combined[labels == 1]

    train_data_tensor = torch.tensor(np.array(selected_data), dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(np.array(not_selected_data), dtype=torch.float32).to(device)

    return train_data_tensor,test_data_tensor


if __name__ == '__main__':
    normal, abnormal = get_data_1d_uci_all_data(20)
    print()
