import glob
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from prototype import constant
from prototype.global_tramsform import transform_sensor_data_to_df
from utils.slidewindow import slide_window2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flag_if_show_image = False
scaler = MinMaxScaler(feature_range=(-1, 1))

def get_mh_data(slide_window_length):
    file_list = glob.glob('../data/mHealth/mHealth_*.log')
    final_data = []
    appended_data = []

    for file_name in file_list:
        print(file_name)
        data = pd.read_csv(file_name,sep='\t',header=None)

        data.columns = ['chest_x', 'chest_y', 'chest_z',
                        'electrocardiogram_1', 'electrocardiogram_2',
                        'ankle_x', 'ankle_y', 'ankle_z',
                        'gyro_x', 'gyro_y', 'gyro_z',
                        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                        'arm_x', 'arm_y', 'arm_z',
                        'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
                        'magnetometer_arm_x', 'magnetometer_arm_y', 'magnetometer_arm_z',
                        'label']
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    record_diff = []
    pre_val = -1
    for index, value in big_df['label'].items():
        if value != pre_val:
            record_diff.append(index)
        pre_val = value

    sliced_list = []
    for i in range(1, len(record_diff)):
        start = record_diff[i-1]
        end = record_diff[i]
        sliced_df = big_df.iloc[start:end]
        if sliced_df['label'].array[0] != 0:
            sliced_list.append(sliced_df)

    for df in sliced_list:
        # 滑动窗口平均噪声
        df['chest_x'] = df['chest_x'].rolling(window=3).mean().bfill()
        df['chest_y'] = df['chest_y'].rolling(window=3).mean().bfill()
        df['chest_z'] = df['chest_z'].rolling(window=3).mean().bfill()

        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(df.to_numpy(), slide_window_length, 0.5)

        # data_processed = []
        # # 对于每个样本组,100条数据,都进行特征合并操作
        # for data_simple in data_sliced_list:
        #     x_axis = data_simple[:, 0]
        #     y_axis = data_simple[:, 1]
        #     z_axis = data_simple[:, 2]
        #     xyz_axis = np.sqrt(x_axis ** 2 + y_axis ** 2 + z_axis ** 2).reshape(-1, 1)
        #     # xyz_axis = merge_data_to_1D(x=x_axis, y=y_axis, z=z_axis, method='fft').reshape(-1, 1)
        #
        #     result = np.hstack((data_simple, xyz_axis))
        #     data_processed.append(result)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)
        print(f'Total number of files: {len(file_list)}, now is No. {file_list.index(file_name)}')

    # shuffle data
    random.shuffle(final_data)
    # 提取输入和标签
    x = np.array([arr[:, 0] for arr in final_data])
    y = np.array([arr[:, 1] for arr in final_data])
    z = np.array([arr[:, 2] for arr in final_data])
    xyz = np.stack((x, y, z), axis=1)

    # 提取输入和标签
    # inputs_tensor = torch.tensor(xyz, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
    # input_features = np.array([arr[:, 24] for arr in final_data])
    inputs_tensor = torch.tensor(xyz, dtype=torch.float32)  # 添加通道维度
    labels = np.array([arr[:, 23] for arr in final_data])[:, 0]
    # 将NumPy数组转换为Tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 计算分割点 7:3
    split_point = int(0.7 * len(inputs_tensor))

    # train data/label   test data/label
    train_data = inputs_tensor[:split_point].to(device)
    test_data = inputs_tensor[split_point:].to(device)
    train_labels = labels_tensor[:split_point].to(device)
    test_labels = labels_tensor[split_point:].to(device)

    return train_data, train_labels, test_data, test_labels

# mHealth 3维度加速度数据
def get_mh_data_1d_3ch(slide_window_length):
    file_list = glob.glob('../data/mHealth/mHealth_*.log')
    final_data = []
    appended_data = []

    for file_name in file_list:
        print(file_name)
        data = pd.read_csv(file_name,sep='\t',header=None)

        data.columns = ['chest_x', 'chest_y', 'chest_z',
                        'electrocardiogram_1', 'electrocardiogram_2',
                        'ankle_x', 'ankle_y', 'ankle_z',
                        'gyro_x', 'gyro_y', 'gyro_z',
                        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                        'arm_x', 'arm_y', 'arm_z',
                        'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
                        'magnetometer_arm_x', 'magnetometer_arm_y', 'magnetometer_arm_z',
                        'label']
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    record_diff = []
    pre_val = -1
    for index, value in big_df['label'].items():
        if value != pre_val:
            record_diff.append(index)
        pre_val = value

    sliced_list = []
    for i in range(1, len(record_diff)):
        start = record_diff[i-1]
        end = record_diff[i]
        sliced_df = big_df.iloc[start:end]
        if sliced_df['label'].array[0] != 0:
            sliced_list.append(sliced_df)

    for df in sliced_list:
        # 滑动窗口平均噪声
        df['chest_x'] = df['chest_x'].rolling(window=3).mean().bfill()
        df['chest_y'] = df['chest_y'].rolling(window=3).mean().bfill()
        df['chest_z'] = df['chest_z'].rolling(window=3).mean().bfill()

        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(df.to_numpy(), slide_window_length, 0.5)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)
        print(f'Total number of files: {len(file_list)}, now is No. {file_list.index(file_name)}')

    # shuffle data
    random.shuffle(final_data)
    # 提取输入和标签
    # 提取输入和标签
    x = np.array([arr[:, 0] for arr in final_data])
    y = np.array([arr[:, 1] for arr in final_data])
    z = np.array([arr[:, 2] for arr in final_data])
    labels = np.array([arr[:, 23] for arr in final_data])[:, 0]
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

def get_mh_data_1d_9ch(slide_window_length, features_num):
    file_list = glob.glob('../data/mHealth/mHealth_*.log')
    final_data = []
    appended_data = []

    for file_name in file_list:
        print(file_name)
        data = pd.read_csv(file_name,sep='\t',header=None)

        data.columns = constant.Constant.mHealth.data_columns
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

    # forearm data
    big_df = big_df.iloc[:, 14:24]

    # Global Transformed
    big_df = transform_sensor_data_to_df(big_df)

    # 归一化
    # big_df.iloc[:, :features_num] = scaler.fit_transform(big_df.iloc[:, :features_num])

    record_diff = []
    pre_val = -1
    for index, value in big_df['label'].items():
        if value != pre_val:
            record_diff.append(index)
        pre_val = value

    sliced_list = []
    for i in range(1, len(record_diff)):
        start = record_diff[i-1]
        end = record_diff[i]
        sliced_df = big_df.iloc[start:end]
        if sliced_df['label'].array[0] != 0:
            sliced_list.append(sliced_df)

    for df in sliced_list:
        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(df.to_numpy(), slide_window_length, 0.5)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)

    # shuffle data
    random.shuffle(final_data)

    # 提取输入和标签
    features = np.array([arr[:, :features_num] for arr in final_data])
    labels = np.array([arr[:, 9] for arr in final_data])[:, 0]

    # 将NumPy数组转换为Tensor
    inputs_tensor = torch.tensor(features, dtype=torch.float32)  # 添加通道维度
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 计算分割点 7:3
    split_point = int(0.7 * len(inputs_tensor))

    # train data/label   test data/label
    train_data = inputs_tensor[:split_point].to(device)
    test_data = inputs_tensor[split_point:].to(device)
    train_labels = labels_tensor[:split_point].to(device)
    test_labels = labels_tensor[split_point:].to(device)

    return train_data, train_labels, test_data, test_labels



# 使用别的数据集,在 mhealth数据集上面进行测试
# walking waiting running
def get_mh_data_1d_3ch_for_test(slide_window_length):
    file_list = glob.glob('../data/mHealth/mHealth_*.log')
    final_data = []
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name, sep='\t', header=None)

        data.columns = ['chest_x', 'chest_y', 'chest_z',
                        'electrocardiogram_1', 'electrocardiogram_2',
                        'ankle_x', 'ankle_y', 'ankle_z',
                        'gyro_x', 'gyro_y', 'gyro_z',
                        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                        'arm_x', 'arm_y', 'arm_z',
                        'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
                        'magnetometer_arm_x', 'magnetometer_arm_y', 'magnetometer_arm_z',
                        'label']

        # filter 1:standing 4:walking 11:running
        filtered_df = data[data['label'].isin([1,4,11])]
        filtered_df.loc[:, 'label'] = filtered_df['label'].replace({1: 2, 4: 1, 11: 3})

        appended_data.append(filtered_df)

    big_df = pd.concat(appended_data, ignore_index=True)
    record_diff = []
    pre_val = -1
    for index, value in big_df['label'].items():
        if value != pre_val:
            record_diff.append(index)
        pre_val = value

    sliced_list = []
    for i in range(1, len(record_diff)):
        start = record_diff[i - 1]
        end = record_diff[i]
        sliced_df = big_df.iloc[start:end]
        if sliced_df['label'].array[0] != 0:
            sliced_list.append(sliced_df)

    for df in sliced_list:
        # 滑动窗口平均噪声
        df.loc[:, 'chest_x'] = df['chest_x'].rolling(window=3).mean().bfill()
        df.loc[:, 'chest_y'] = df['chest_y'].rolling(window=3).mean().bfill()
        df.loc[:, 'chest_z'] = df['chest_z'].rolling(window=3).mean().bfill()

        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(df.to_numpy(), slide_window_length, 0.5)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)

    # shuffle data
    random.shuffle(final_data)

    # 提取XYZ加速度,合成一个三维向量, 提取标签
    x = np.array([arr[:, 0].astype(np.float64) for arr in final_data])
    y = np.array([arr[:, 1].astype(np.float64) for arr in final_data])
    z = np.array([arr[:, 2].astype(np.float64) for arr in final_data])
    xyz = np.stack((x, y, z), axis=1)
    labels = np.array([arr[:, 23] for arr in final_data])[:, 0]

    # 提取数据和标签
    data = torch.tensor(xyz, dtype=torch.float32).to(device)
    label = torch.tensor(labels, dtype=torch.long).to(device)

    return data, label

# 获取mHealth异常数据,把walk作为正常,walk以外为异常进行检测
def get_mh_data_for_abnormal_test(slide_window_length, features_num):
    file_list = glob.glob('../data/mHealth/mHealth_*.log')
    final_data = []
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name, sep='\t', header=None)

        data.columns = ['chest_x', 'chest_y', 'chest_z',
                        'electrocardiogram_1', 'electrocardiogram_2',
                        'ankle_x', 'ankle_y', 'ankle_z',
                        'gyro_x', 'gyro_y', 'gyro_z',
                        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
                        'arm_x', 'arm_y', 'arm_z',
                        'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
                        'magnetometer_arm_x', 'magnetometer_arm_y', 'magnetometer_arm_z',
                        'label']

        # filter 1:standing 4:walking 11:running
        # filtered_df = data[data['label'].isin([1, 4, 11])]
        # filtered_df.loc[:, 'label'] = filtered_df['label'].replace({1: 2, 4: 1, 11: 3})

        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

    # 归一化
    # big_df.iloc[:, :23] = scaler.fit_transform(big_df.iloc[:, :23])

    record_diff = []
    pre_val = -1
    for index, value in big_df['label'].items():
        if value != pre_val:
            record_diff.append(index)
        pre_val = value

    sliced_list = []
    for i in range(1, len(record_diff)):
        start = record_diff[i - 1]
        end = record_diff[i]
        sliced_df = big_df.iloc[start:end]
        if sliced_df['label'].array[0] != 0:
            sliced_list.append(sliced_df)

    for df in sliced_list:
        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(df.to_numpy(), slide_window_length, 0.5)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)

    # shuffle data
    random.shuffle(final_data)
    data = np.array([arr[:, :].astype(np.float64) for arr in final_data])

    # 根据标签,分割数据
    # 1.0 STANDING 4.0 WALK 5.0 upstair
    condition = data[:, :, 23] != 5.0
    condition_abnormal = data[:, :, 23] == 5.0

    # 使用布尔索引进行分割
    tensor_walk = data[condition[:, 0]]
    tensor_not_walk = data[condition_abnormal[:, 0]]

    tensor_walk_tensor = torch.tensor(tensor_walk, dtype=torch.float32).to(device)

    tensor_not_walk_tensor = torch.tensor(tensor_not_walk, dtype=torch.float32).to(device)

    return tensor_walk_tensor[:, :, :features_num], tensor_not_walk_tensor[:, :, :features_num]

if __name__ == '__main__':
    t, f = get_mh_data_for_abnormal_test(128, 6)
    print(t)