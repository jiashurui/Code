import glob
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from prototype.constant import Constant
from utils.slidewindow import slide_window2

stop_simple = 500  # 数据静止的个数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化 MinMaxScaler(Normalization [-1,1])
scaler = MinMaxScaler(feature_range=(-1, 1))

def get_realworld_for_abnormal(slide_window_length):
    # 创建示例输入数据 TODO 这里只用waist做实验, UCI是waist(腰部),mHealth是chest(胸部)
    file_list = glob.glob('../../data/realworld/*/acc_*_waist.csv')
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
        data.iloc[:, :3] = scaler.fit_transform(data.iloc[:, :3])

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
    data_tensor = torch.tensor(input_features, dtype=torch.float32)  # 添加通道维度

    # 根据标签,分割数据
    condition = data_tensor[:, :, 3] == 7.0  # walking
    stand_condition = data_tensor[:, :, 3] == 6.0  # standing

    # 使用布尔索引进行分割
    tensor_walk = data_tensor[condition[:, 0]]
    tensor_not_walk = data_tensor[stand_condition[:, 0]]

    return tensor_walk[:, :, :3], tensor_not_walk[:, :, :3]

if __name__ == '__main__':
    walk,stand = get_data(128)
    print(walk.shape)