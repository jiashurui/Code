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
    file_list = glob.glob('../data/realworld/1/acc_*.csv')
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

        # print(f'before{data['attr_x']}')
        # 滑动窗口平均噪声
        data['attr_x'] = data['attr_x'].rolling(window=3).mean().bfill()
        # print(f'after{data['attr_x']}')

        # 特征合并
        # data['xyz'] = data.apply(lambda row:
        #                          np.sqrt(row['attr_x'] ** 2 + row['attr_y'] ** 2 + row['attr_z'] ** 2)
        #                          , axis=1)
        # merge_data_to_1D(data,'fft')

        # show_me_data1(data[1000:1100], ['attr_x','attr_y','attr_z','xyz'])

        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(data.to_numpy(), slide_window_length, 0.5)

        data_processed = []
        # 对于每个样本组,100条数据,都进行特征合并操作
        for data_simple in data_sliced_list:
            x_axis = data_simple[:, 2]
            y_axis = data_simple[:, 3]
            z_axis = data_simple[:, 4]
            # xyz_axis = np.sqrt(x_axis ** 2 + y_axis ** 2 + z_axis ** 2).reshape(-1, 1)
            xyz_axis = merge_data_to_1D(x=x_axis, y=y_axis, z=z_axis, method='fft').reshape(-1, 1)

            result = np.hstack((data_simple, xyz_axis))
            data_processed.append(result)

        # show_me_data2(data_sliced,['attr_x','attr_y','attr_z','xyz'])

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_processed)
        print(f'Total number of files: {len(file_list)}, now is No. {file_list.index(file_name)}')

    # shuffle data
    random.shuffle(final_data)
    # 提取输入和标签
    input_features = np.array([arr[:, 6] for arr in final_data])
    labels = np.array([arr[:, 5] for arr in final_data])[:, 0]

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


def merge_data_to_1D(x, y, z, method):
    if method == 'fft':
        window = np.hamming(len(x))

        # x fft
        fft_x_origin = x * window
        fft_x_no_dc = fft_x_origin - np.mean(fft_x_origin)
        fft_x_result = np.fft.fft(fft_x_no_dc)
        fft_x_magnitude = np.abs(fft_x_result)

        # y fft
        fft_y_origin = y * window
        fft_y_no_dc = fft_y_origin - np.mean(fft_y_origin)
        fft_y_result = np.fft.fft(fft_y_no_dc)
        fft_y_magnitude = np.abs(fft_y_result)

        # z fft
        fft_z_origin = z * window
        fft_z_no_dc = fft_z_origin - np.mean(fft_z_origin)
        fft_z_result = np.fft.fft(fft_z_no_dc)
        fft_z_magnitude = np.abs(fft_z_result)

        # 对信号进行标准化
        fft_x_magnitude = (fft_x_magnitude - np.mean(fft_x_magnitude)) / np.std(fft_x_magnitude)
        fft_y_magnitude = (fft_y_magnitude - np.mean(fft_y_magnitude)) / np.std(fft_y_magnitude)
        fft_z_magnitude = (fft_z_magnitude - np.mean(fft_z_magnitude)) / np.std(fft_z_magnitude)

        # merge xyz
        fft_xyz_magnitude = np.sqrt(fft_x_magnitude ** 2 + fft_y_magnitude ** 2 + fft_z_magnitude ** 2)

        return fft_xyz_magnitude
