import glob
import random

import numpy as np
import pandas as pd
import torch

from prototype.constant import Constant
from utils.slidewindow import slide_window2

stop_simple = 500  # 数据静止的个数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(slide_window_length):
    # 创建示例输入数据 TODO
    file_list = glob.glob('../data/realworld/*/acc_*.csv')
    final_data = []

    # make label by fileName (walking)
    # chest 1 forearm 2 head 3 shin 4 thigh 5 upper arm 6 waist 7
    label_map = Constant.RealWorld.label_map
    for file_name in file_list:
        data = pd.read_csv(file_name)
        # 对于每一个dataframe , 按照文件名给其打上标签
        matched_substrings = [label for label in label_map.keys() if label in file_name]

        if not matched_substrings or len(matched_substrings) != 1:
            raise KeyError("无法生成标签")
        else:
            data['label'] = label_map.get(matched_substrings[0])
        ########################################################
        # 按照行处理数据
        # 数据处理(合并特征到1维度)
        # TODO 判断初始手机朝向, 数据转换(暂时先试试不用转换能不能做)

        # 去除头部
        data = data[stop_simple: len(data)]

        # 滑动窗口平均噪声
        data.rolling(window=3).mean()

        # 特征合并
        data['xyz'] = data.apply(lambda row:
                                 np.sqrt(row['attr_x'] ** 2 + row['attr_y'] ** 2 + row['attr_z'] ** 2)
                                 , axis=1)

        # show_me_data1(data[1000:1100], ['attr_x','attr_y','attr_z','xyz'])

        # 分割后的数据 100个 X组
        data_sliced = slide_window2(data, slide_window_length, 0.5)

        # show_me_data2(data_sliced,['attr_x','attr_y','attr_z','xyz'])
        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced)
        print(f'Total number of files: {len(file_list)}, now is No. {file_list.index(file_name)}')

    # shuffle data
    random.shuffle(final_data)
    # 提取输入和标签
    input_features = np.array([df['xyz'].values for df in final_data])
    labels = np.array([df['label'].values for df in final_data])[:, 0]

    # 将NumPy数组转换为Tensor
    inputs_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 计算分割点 7:3
    split_point = int(0.7 * len(inputs_tensor))

    # train data/label   test data/label
    train_data = inputs_tensor[:split_point].to(device)
    test_data = inputs_tensor[split_point:].to(device)
    train_labels = labels_tensor[:split_point].to(device)
    test_labels = labels_tensor[split_point:].to(device)

    return train_data, train_labels, test_data, test_labels
