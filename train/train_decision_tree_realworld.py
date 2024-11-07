import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from datareader.mh_datareader import simple_get_mh_all_features
from datareader.realworld_datareader import simple_get_realworld_all_features
from prototype.constant import Constant
from statistic.stat_common import calc_df_features, calc_fft_spectral_energy, spectral_entropy, calc_acc_sma, \
    spectral_centroid, dominant_frequency, calculate_ar_coefficients
from utils.dict_utils import find_key_by_value

# K = 8 に設定する
K = 8
simpling = 50
features_number = 9
slice_length = 256
# 全局变换之后RealWorld数据(全局变换按照frame进行)
origin_data = simple_get_realworld_all_features(slice_length, type='df', with_rpy= True)
origin_data_np = np.array(origin_data)

features_list = []
for d in origin_data:
    df_features, _ = calc_df_features(d.iloc[:, :9])

    # 分别对9维数据XYZ求FFT的能量(结果会变坏)
    aex,aey,aez,aet = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='acc_attr_x', acc_y_name='acc_attr_y', acc_z_name='acc_attr_z', T=simpling)
    gex,gey,gez,get = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='gyro_attr_x', acc_y_name='gyro_attr_y', acc_z_name='gyro_attr_z', T=simpling)
    mex,mey,mez,met = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='mag_attr_x', acc_y_name='mag_attr_y', acc_z_name='mag_attr_z', T=simpling)
    df_features['fft_spectral_energy'] = [aex,aey,aez,gex,gey,gez,mex,mey,mez]

    # 分别对9维数据XYZ求FFT的能量(结果会变坏)
    aex,aey,aez,aet = spectral_entropy(d.iloc[:, :9], acc_x_name='acc_attr_x', acc_y_name='acc_attr_y', acc_z_name='acc_attr_z', T=simpling)
    gex,gey,gez,get = spectral_entropy(d.iloc[:, :9], acc_x_name='gyro_attr_x', acc_y_name='gyro_attr_y', acc_z_name='gyro_attr_z', T=simpling)
    mex,mey,mez,met = spectral_entropy(d.iloc[:, :9], acc_x_name='mag_attr_x', acc_y_name='mag_attr_y', acc_z_name='mag_attr_z', T=simpling)
    df_features['fft_spectral_entropy'] = [aex,aey,aez,gex,gey,gez,mex,mey,mez]

    centroid_arr = []
    dominant_frequency_arr = []
    ar_co_arr = []
    for i in (range(features_number)):
        centroid_feature = spectral_centroid(d.iloc[:, i].values, sampling_rate=10)
        dominant_frequency_feature = dominant_frequency(d.iloc[:, i].values, sampling_rate=10)
        ar_coefficients = calculate_ar_coefficients(d.iloc[:, i].values)

        centroid_arr.append(centroid_feature)
        dominant_frequency_arr.append(dominant_frequency_feature)
        ar_co_arr.append(ar_coefficients)

    df_features['fft_spectral_centroid'] = np.array(centroid_arr)
    df_features['fft_dominant_frequency'] = np.array(dominant_frequency_arr)

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

train_data = np.array(features_list)


# 假设你已经训练好了一个决策树模型
clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)

# 保存模型