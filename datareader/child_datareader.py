import glob
import random

import numpy as np
import pandas as pd
import torch

from prototype.constant import Constant
from utils.slidewindow import slide_window2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_child_data_rnn(slide_window_length):
    labeled_path = '/Users/jiashurui/Desktop/Dataset_labeled/merged_data/*.csv'

    file_list = glob.glob(labeled_path)
    final_data = []
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    record_diff = []
    pre_val = -1
    for index, value in big_df['X'].items():
        if value != pre_val:
            record_diff.append(index)
        pre_val = value

    sliced_list = []
    for i in range(1, len(record_diff)):
        start = record_diff[i - 1]
        end = record_diff[i]
        sliced_df = big_df.iloc[start:end]
        if sliced_df['X'].array[0] != 'なし':
            sliced_list.append(sliced_df)

    for df in sliced_list:
        # 滑动窗口平均噪声
        df.loc[:, 'accx'] = df['accx'].rolling(window=3).mean().bfill()
        df.loc[:, 'accy'] = df['accy'].rolling(window=3).mean().bfill()
        df.loc[:, 'accz'] = df['accz'].rolling(window=3).mean().bfill()

        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(df.to_numpy(), slide_window_length, 0.5)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)

    # shuffle data
    random.shuffle(final_data)

    ########    ########    ########    ########    ########    ########    ########
    # 提取输入和标签
    x = np.array([arr[:, 1].astype(np.float64) for arr in final_data])
    y = np.array([arr[:, 2].astype(np.float64) for arr in final_data])
    z = np.array([arr[:, 3].astype(np.float64) for arr in final_data])

    labels = np.array([arr[:, 23] for arr in final_data])[:, 0]
    mapping = Constant.ChildWalk.action_map
    labels = np.array([mapping.get(item, '0') for item in labels])

    xyz = np.stack((x, y, z), axis=1)

    # 将NumPy数组转换为Tensor
    inputs_tensor = torch.tensor(xyz, dtype=torch.float32)  # 添加通道维度
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 计算分割点 7:3
    split_point = int(0.7 * len(inputs_tensor))

    # train data/label   test data/label
    train_data = inputs_tensor[:split_point].to(device)
    test_data = inputs_tensor[split_point:].to(device)
    train_labels = labels_tensor[:split_point].to(device)
    test_labels = labels_tensor[split_point:].to(device)

    return train_data, train_labels, test_data, test_labels