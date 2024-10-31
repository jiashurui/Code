import glob
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from prototype.global_tramsform import transform_sensor_data_to_df
from utils.config_utils import get_value_from_config
from utils.slidewindow import slide_window2

# 2024年7月26日 研究室 5名大学生行动数据

# 行動順：　１、立つ
# 　　　　　２、フラフラ
# 　　　　　３、しゃがむ
# 　　　　　４、飛ぶ
# 　　　　　５、歩く
# 　　　　　６、走る

# １　賈：　　15:19 ~ 15:25 　　   1721974740000 ~ 1721975100000
# ２　溝脇：　15:27 ~ 15:33        1721975220000 ~ 1721975580000
# ３　高野：　15:34 ~ 15:40        1721975640000 ~ 1721976000000
# ４　高橋：　15:41 ~ 15:47        1721976060000 ~ 1721976420000
# ５　金：　　15:48 ~ 15:54        1721976480000 ~ 1721976840000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化 MinMaxScaler
scaler = MinMaxScaler()


def get_stu_data(slide_window_length):
    base_path = get_value_from_config('stu_data_set')

    acc_file = glob.glob(base_path)
    final_data = []
    df = pd.read_csv(acc_file[0])

    # filter
    # df = data[data['label'] != -1]

    record_diff = []
    pre_val = -2
    # 按照label,分成各个label单位的小组
    for index, value in df['label'].items():
        if value != pre_val:
            record_diff.append(index)
        pre_val = value

    sliced_list = []
    for i in range(1, len(record_diff) - 1):
        start = record_diff[i]
        end = record_diff[i + 1]
        sliced_df = df.iloc[start:end]

        if sliced_df['label'].array[0] != -1:
            sliced_list.append(sliced_df)

    for df in sliced_list:
        # 滑动窗口平均噪声
        df.loc[:, 'x(m/s2)'] = df['x(m/s2)'].rolling(window=3).mean().bfill()
        df.loc[:, 'y(m/s2)'] = df['y(m/s2)'].rolling(window=3).mean().bfill()
        df.loc[:, 'z(m/s2)'] = df['z(m/s2)'].rolling(window=3).mean().bfill()

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

    labels = np.array([arr[:, 4] for arr in final_data])[:, 0]
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


# 获取长冈科技大学,学生所有特征
def get_stu_all_features(slide_window_length, option='with_label'):
    base_path = '../data/student/0726_lab/merge_labeled.csv'

    acc_file = glob.glob(base_path)
    final_data = []
    df = pd.read_csv(acc_file[0])

    # 对 DataFrame 的每一列进行归一化
    # df.iloc[:, 1:10] = scaler.fit_transform(df.iloc[:, 1:10])

    record_diff = []
    pre_val = -2
    # 按照label,分成各个label单位的小组
    for index, value in df['label'].items():
        if value != pre_val:
            record_diff.append(index)
        pre_val = value

    sliced_list = []
    for i in range(1, len(record_diff) - 1):
        start = record_diff[i]
        end = record_diff[i + 1]
        sliced_df = df.iloc[start:end]

        if sliced_df['label'].array[0] != -1:
            sliced_list.append(sliced_df)

    for df in sliced_list:
        # 分割后的数据 100个 X组
        # 全局变换
        df = transform_sensor_data_to_df(df)

        # 滑动窗口分割
        data_sliced_list = slide_window2(df.to_numpy(), slide_window_length, 0.5)

        # 对于每一个dataframe , 滑动窗口分割数据
        final_data.extend(data_sliced_list)

    # shuffle data
    random.shuffle(final_data)

    if option == 'with_label':
        # 提取输入和标签
        data = np.array([arr[:, 1:11].astype(np.float64) for arr in final_data])

        # 将NumPy数组转换为Tensor
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        return data_tensor
    elif option == 'without_label':
        # 提取输入和标签
        data = np.array([arr[:, 1:10] for arr in final_data])
        label = np.array([arr[:, 10] for arr in final_data])[:, 0]

        # 将NumPy数组转换为Tensor
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        label_tensor = torch.tensor(label, dtype=torch.long).to(device)

        return data_tensor, label_tensor


# 简单地获取一些特征(对比上面的, 活动区分没有那么精确)
def simple_get_stu_all_features(slide_window_length, type='tensor', filtered_label=[], mapping_label={}):
    file = glob.glob('../data/student/0726_lab/merge_labeled.csv')
    df = pd.read_csv(file[0])

    # 按照int读取数据
    df['label'] = df['label'].astype(int)
    df = df[df['label'] != -1]

    if filtered_label:
        df = df[~df['label'].isin(filtered_label)]
        df['label'] = df['label'].map(mapping_label)

    df = df.iloc[:, 1:11]
    # 全局变换
    df = transform_sensor_data_to_df(df)
    df_list = slide_window2(df, slide_window_length, 0.5)

    transformed_list = []
    for d in df_list:
        transformed_frame = transform_sensor_data_to_df(d)
        transformed_list.append(transformed_frame)
    np_arr = np.array(transformed_list)
    data_tensor = torch.tensor(np_arr, dtype=torch.float32).to(device)

    if type == 'df':
        return transformed_list
    elif type == 'np':
        return np_arr

    return data_tensor


# 传递出所有特征(不带标签)
def get_stu_part_features(slide_window_length, feature_num, label_for_abnormal_test):
    all_features_data = get_stu_all_features(slide_window_length)

    # 根据标签,分割数据
    condition = all_features_data[:, :, 9] == label_for_abnormal_test

    # 使用布尔索引进行分割
    tensor_train = all_features_data[~condition[:, 0]]  # 不满足条件
    tensor_test = all_features_data[condition[:, 0]]  # 满足条件

    # TODO long lat
    return tensor_train[:, :, :feature_num], tensor_test[:, :, :feature_num]


if __name__ == '__main__':
    normal = get_stu_all_features(20)
    print()
