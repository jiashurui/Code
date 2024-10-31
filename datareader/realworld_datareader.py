import glob
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from prototype.constant import Constant
from prototype.global_tramsform2 import transform_sensor_data_to_df2
from utils.show import show_acc_data_before_transformed
from utils.slidewindow import slide_window2
from .convert_common import convert_df_columns_name

stop_simple = 500  # 数据静止的个数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化 MinMaxScaler(Normalization [-1,1])
scaler = StandardScaler()

def read_data():
    file_name = '../data/realworld/*/shin_merged.csv'
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    df = pd.concat(appended_data, ignore_index=True)

    # 进行数据筛选, 仅仅选择步行数据
    df = df[df['label'] == 7]

    # 进行列名转换
    df = convert_df_columns_name(df)

    # Global Transform
    df , df_transformed = transform_sensor_data_to_df2(df)

    # 数据展示
    show_acc_data_before_transformed(df , df_transformed,300,400)

    return df


def get_realworld_for_abnormal(slide_window_length):
    # 创建示例输入数据 TODO 这里只用waist做实验, UCI是waist(腰部),mHealth是chest(胸部)
    file_list = glob.glob('../data/realworld/*/acc_*_waist.csv')
    final_data = []

    # make label by fileName (walking)
    # chest 1 forearm 2 head 3 shin 4 thigh 5 upper arm 6 waist 7
    label_map = Constant.RealWorld.action_map
    for file_name in file_list:
        data = pd.read_csv(file_name)

        # 对于每一个dataframe , 按照文件名给其打上标签
        matched_substrings = [label for label in label_map.keys() if label in file_name]

        if not matched_substrings or len(matched_substrings) != 1:
            raise KeyError("无法生成标签")
        else:
            data['label'] = label_map.get(matched_substrings[0])
        ########################################################

        # 去除头部
        data = data[stop_simple: len(data)]

        # 去除不要的数据(时间和ID)
        data = data.iloc[:, 2:]

        # 归一化
        # data.iloc[:, :3] = scaler.fit_transform(data.iloc[:, :3])

        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(data.to_numpy(), slide_window_length, 0.5)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)
        print(f'Total number of files: {len(file_list)}, now is No. {file_list.index(file_name)}')

    # shuffle data
    random.shuffle(final_data)

    # 提取输入和标签
    input_features = np.array([arr[:, :4] for arr in final_data])

    # 将NumPy数组转换为Tensor
    data_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)  # 添加通道维度

    # 根据标签,分割数据
    condition = data_tensor[:, :, 3] != 6.0  # standing
    stand_condition = data_tensor[:, :, 3] == 6.0  # standing

    # 使用布尔索引进行分割
    tensor_walk = data_tensor[condition[:, 0]]
    tensor_not_walk = data_tensor[stand_condition[:, 0]]

    return tensor_walk[:, :, :3], tensor_not_walk[:, :, :3]


# 读取Realworld数据用于训练模型
def get_realworld_for_recon(slide_window_length, features_num, filtered_label=[], mapping_label={}):
    # 创建示例输入数据
    file_list = glob.glob('../data/realworld/*/forearm_merged.csv')
    final_data = []
    for file_name in file_list:
        data = pd.read_csv(file_name)

        # 标签按照整数读取
        data['label'] = data['label'].astype(int)

        # 去除头部 (Realworld 特有, 每段大数据前面有停止一段时间)
        data = data[stop_simple: len(data)]

        # 过滤指定标签数据
        if filtered_label:
            data = data[~data['label'].isin(filtered_label)]
            data['label'] = data['label'].map(mapping_label)

        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(data, slide_window_length, 0.5)

        # 对每一个时间片进行处理
        transformed_list = []
        for d in data_sliced_list:
            # 全局转换
            transformed_frame = transform_sensor_data_to_df2(d)
            # 归一化
            transformed_frame.iloc[:, :9] = scaler.fit_transform(transformed_frame.iloc[:, :9])

            transformed_list.append(transformed_frame)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(transformed_list)
        print(f'Total number of files: {len(file_list)}, now is No. {file_list.index(file_name)}')

    # shuffle data
    random.shuffle(final_data)

    # 提取输入和标签
    input_features = np.array([arr.iloc[:, :features_num] for arr in final_data])
    labels = np.array([arr.iloc[:, 9] for arr in final_data])[:, 0]

    # 将NumPy数组转换为Tensor
    data_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
    data_label = torch.tensor(labels, dtype=torch.long).to(device)

    # 计算分割点 7:3
    split_point = int(0.7 * len(data_tensor))

    # train data/label   test data/label
    train_data = data_tensor[:split_point].to(device)
    test_data = data_tensor[split_point:].to(device)
    train_labels = data_label[:split_point].to(device)
    test_labels = data_label[split_point:].to(device)

    return train_data,train_labels,test_data,test_labels


# 读取realworld数据(不做任何处理变换)
def get_realworld_raw_for_abnormal(slide_window_length, features_num, global_transform=False):
    # 创建示例输入数据 TODO 这里只用waist做实验, UCI是waist(腰部),mHealth是chest(胸部)
    file_list = glob.glob('../data/realworld/*/waist_merged.csv')
    final_data = []
    for file_name in file_list:
        data = pd.read_csv(file_name)

        # 去除头部
        data = data[stop_simple: len(data)]

        if global_transform:
            data = transform_sensor_data_to_df2(data)
        # 归一化 TODO 2024/10/25 归一化会影响重建
        # data.iloc[:, :9] = scaler.fit_transform(data.iloc[:, :9])

        # 分割后的数据 100个 X组
        data_sliced_list = slide_window2(data.to_numpy(), slide_window_length, 0.5)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)
        print(f'Total number of files: {len(file_list)}, now is No. {file_list.index(file_name)}')

    # shuffle data
    random.shuffle(final_data)

    # 提取输入和标签
    input_features = np.array([arr[:, :] for arr in final_data])
    labels = np.array([arr[:, 9] for arr in final_data])[:, 0]

    # 将NumPy数组转换为Tensor
    data_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
    data_label = torch.tensor(labels, dtype=torch.long).to(device)

    # 根据标签,分割数据
    condition = data_tensor[:, :, 9] == 7.0  # walking
    stand_condition = data_tensor[:, :, 9] != 7.0  # not walking

    # 使用布尔索引进行分割
    tensor_walking = data_tensor[condition[:, 0]]
    tensor_not_walking = data_tensor[stand_condition[:, 0]]

    # 计算分割点 7:3
    split_point = int(0.7 * len(tensor_walking))

    # train data/label   test data/label
    train_normal_data = tensor_walking[:split_point].to(device)
    test_abnormal_data = tensor_walking[split_point:].to(device)

    return train_normal_data[:, :, :features_num], test_abnormal_data[:, :, :features_num], tensor_not_walking[:, :, :features_num]


# 读取realworld数据用于异常检测
def get_realworld_transformed_for_abnormal(slide_window_length, features_num):
    return get_realworld_raw_for_abnormal(slide_window_length, features_num, global_transform=True)

if __name__ == '__main__':
    normal,abnormal,test_abnormal = get_realworld_transformed_for_abnormal(128, 6)
    print()