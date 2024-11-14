import glob
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from filter.filter import butter_lowpass_filter
from prototype.global_tramsform import transform_sensor_data_to_df, transform_sensor_data_to_np, \
    transform_sensor_data_to_df1
from statistic.stat_common import spectral_centroid, dominant_frequency, calculate_ar_coefficients, spectral_entropy, \
    calc_fft_spectral_energy, calc_df_features, calc_acc_sma
from utils.config_utils import get_value_from_config
from utils.slidewindow import slide_window2

# 2024年11月11日 研究室 6名大学生行动数据

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
scaler = MinMaxScaler(feature_range=(-1, 1))


# 简单地获取一些特征(对比上面的, 活动区分没有那么精确)
def simple_get_stu_all_features(slide_window_length, type='tensor', filtered_label=[], mapping_label={}, with_rpy=False):
    file = glob.glob('../data/student/1111_lab/merged.csv')
    df = pd.read_csv(file[0])

    # 按照int读取数据
    df['label'] = df['label'].astype(int)

    # 对标签进行filter
    if filtered_label:
        df = df[~df['label'].isin(filtered_label)]
        df['label'] = df['label'].map(mapping_label)

    # 提取数据, 读取数据和标签(去除掉时间)
    df = df.iloc[:, 1:11]

    # 滑动窗口
    df_list = slide_window2(df, slide_window_length, 0.5)

    # random.shuffle(df_list)

    # 对每一个时间片进行处理
    transformed_list = []
    for d in df_list:
        # 低通滤波器
        d = d.apply(lambda x: butter_lowpass_filter(x, 3, 10, 4))

        if with_rpy:
            transformed_frame = transform_sensor_data_to_df1(d)
        else:
            transformed_frame = transform_sensor_data_to_df(d)

        # transformed_frame.iloc[:, :9] = scaler.fit_transform(transformed_frame.iloc[:, :9])
        transformed_list.append(transformed_frame)

    np_arr = np.array(transformed_list)
    data_tensor = torch.tensor(np_arr, dtype=torch.float32).to(device)

    if type == 'df':
        return transformed_list
    elif type == 'np':
        return np_arr

    return data_tensor


# 获取高维度特征
def get_features(origin_data):
    features_list = []
    for d in origin_data:
        df_features, _ = calc_df_features(d.iloc[:, :9])

        # 分别对9维数据XYZ求FFT的能量(结果会变坏)
        aex, aey, aez, aet = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='accx', acc_y_name='accy',
                                                      acc_z_name='accz', T=10)
        gex, gey, gez, get = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='angx', acc_y_name='angy',
                                                      acc_z_name='angz', T=10)
        mex, mey, mez, met = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='magx', acc_y_name='magy',
                                                      acc_z_name='magz', T=10)
        df_features['fft_spectral_energy'] = [aex, aey, aez, gex, gey, gez, mex, mey, mez]

        # 分别对9维数据XYZ求FFT的能量(结果会变坏)
        aex, aey, aez, aet = spectral_entropy(d.iloc[:, :9], acc_x_name='accx', acc_y_name='accy',
                                              acc_z_name='accz', T=10)
        gex, gey, gez, get = spectral_entropy(d.iloc[:, :9], acc_x_name='angx', acc_y_name='angy',
                                              acc_z_name='angz', T=10)
        mex, mey, mez, met = spectral_entropy(d.iloc[:, :9], acc_x_name='magx', acc_y_name='magy', acc_z_name='magz',
                                              T=10)
        df_features['fft_spectral_entropy'] = [aex, aey, aez, gex, gey, gez, mex, mey, mez]

        centroid_arr = []
        dominant_frequency_arr = []
        ar_co_arr = []
        for i in (range(9)):
            centroid_feature = spectral_centroid(d.iloc[:, i].values, sampling_rate=10)
            dominant_frequency_feature = dominant_frequency(d.iloc[:, i].values, sampling_rate=10)
            ar_coefficients = calculate_ar_coefficients(d.iloc[:, i].values)

            centroid_arr.append(centroid_feature)
            dominant_frequency_arr.append(dominant_frequency_feature)
            ar_co_arr.append(ar_coefficients)

        df_features['fft_spectral_centroid'] = np.array(centroid_arr)
        df_features['fft_dominant_frequency'] = np.array(dominant_frequency_arr)
        df_features['ar_coefficients'] = np.array(ar_co_arr)

        # 舍弃掉磁力数据(结果会变坏)
        # df_features = df_features.iloc[:6, :]

        # 特征打平
        flatten_val = df_features.values.flatten()

        # 单独一维特征
        # 加速度XYZ
        acc_sma = calc_acc_sma(d.iloc[:, 0], d.iloc[:, 1], d.iloc[:, 2])
        roll_avg = d.iloc[:, 10].mean()
        pitch_avg = d.iloc[:, 11].mean()
        yaw_avg = d.iloc[:, 12].mean()

        flatten_val = np.append(flatten_val, acc_sma)
        flatten_val = np.append(flatten_val, roll_avg)
        flatten_val = np.append(flatten_val, pitch_avg)
        flatten_val = np.append(flatten_val, yaw_avg)

        features_list.append(flatten_val)
    return features_list

if __name__ == '__main__':
    print()
