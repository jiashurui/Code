import glob
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from utils.slidewindow import slide_window2


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

    data_x = pd.read_csv(file_x, sep='\s+', header=None)
    data_y = pd.read_csv(file_y, sep='\s+', header=None)
    data_z = pd.read_csv(file_z, sep='\s+', header=None)

    index_of_change = get_change_points_excluding_first(file_subject)

    final_list = []

    for i in (range(len(index_of_change)- 1)) :
        start = index_of_change[i]
        end = index_of_change[i+1]

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
    return test_data,labels