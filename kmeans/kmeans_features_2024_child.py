import stat

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datareader.child_datareader import get_child_all_features, simple_get_child_2024_all_features, \
    simple_get_child_2023_all_features
from datareader.datareader_stu import get_stu_all_features, simple_get_stu_all_features
from prototype.constant import Constant
from statistic.stat_common import calc_df_features, calc_acc_sma, spectral_centroid, dominant_frequency, \
    calculate_ar_coefficients, spectral_entropy, calc_fft_spectral_energy
from utils.dict_utils import find_key_by_value

# K = 6 に設定する
K = 3
features_number = 9
slice_length = 20
simpling = 10

# 全局变换之后的小学生数据(全局变换按照frame进行)
# 注意,这里不带标签(9维数据 + rpy)
origin_data = simple_get_child_2024_all_features(slice_length, type= 'df')
origin_data_np = np.array(origin_data)

features_list = []
for d in origin_data:
    df_features, _ = calc_df_features(d.iloc[:, :9])

    # 分别对9维数据XYZ求FFT的能量(结果会变坏)
    aex,aey,aez,aet = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='acc_x', acc_y_name='acc_y', acc_z_name='acc_z', T=simpling)
    gex,gey,gez,get = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='gyro_x', acc_y_name='gyro_y', acc_z_name='gyro_z', T=simpling)
    mex,mey,mez,met = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='mag_x', acc_y_name='mag_y', acc_z_name='mag_z', T=simpling)
    df_features['fft_spectral_energy'] = [aex,aey,aez,gex,gey,gez,mex,mey,mez]

    # 分别对9维数据XYZ求FFT的能量(结果会变坏)
    aex,aey,aez,aet = spectral_entropy(d.iloc[:, :9], acc_x_name='acc_x', acc_y_name='acc_y', acc_z_name='acc_z', T=simpling)
    gex,gey,gez,get = spectral_entropy(d.iloc[:, :9], acc_x_name='gyro_x', acc_y_name='gyro_y', acc_z_name='gyro_z', T=simpling)
    mex,mey,mez,met = spectral_entropy(d.iloc[:, :9], acc_x_name='mag_x', acc_y_name='mag_y', acc_z_name='mag_z', T=simpling)
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
    roll_avg = d.iloc[:, 9].mean()
    pitch_avg = d.iloc[:, 10].mean()
    yaw_avg = d.iloc[:, 11].mean()

    flatten_val = np.append(flatten_val, acc_sma)
    flatten_val = np.append(flatten_val, roll_avg)
    flatten_val = np.append(flatten_val, pitch_avg)
    flatten_val = np.append(flatten_val, yaw_avg)
    features_list.append(flatten_val)

train_data = np.array(features_list)
pca = PCA(n_components=0.7, random_state=3407)
scaler = StandardScaler()
# n_components specifies how many principal components to keep
normal_latent = scaler.fit_transform(train_data)
normal_result = pca.fit_transform(normal_latent)


explained_variance = pca.explained_variance_ratio_
print("PCA 维度:", len(explained_variance))
print("方差解释率:", explained_variance)
print("方差累计解释率:", np.sum(explained_variance))


# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=K, random_state=123)
kmeans.fit(normal_result)

# 获取聚类结果
labels = kmeans.labels_  # 每个样本的聚类标签
centroids = kmeans.cluster_centers_  # 聚类质心


# 6. 使用 numpy 计数每个聚类的样本数量
unique, counts = np.unique(labels, return_counts=True)

# 7. 打印每个聚类的样本数量
print("每个聚类的样本数量：")
for cluster, count in zip(unique, counts):
    print(f"聚类 {cluster}: {count} 个样本")

# 初始化列表用于存储索引
indices = {0: [], 1: [], 2: [],}

# 遍历数据并记录索引
for index, value in enumerate(labels):
    if value in indices:
        indices[value].append(index)


# 3. 随机选择每种索引中的 5 个样本
sample_data = {key: np.random.choice(value, size=5, replace=False) for key, value in indices.items()}

# 4. 打印和可视化
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 10))

for i, (key, samples) in enumerate(sample_data.items()):
    for j, sample_index in enumerate(samples):
        sample_data = origin_data_np[sample_index]

        # 绘制折线图
        axes[i, j].plot(sample_data[:, 0], label='Feature 0', color='r')  # 第 0 个特征
        axes[i, j].plot(sample_data[:, 1], label='Feature 1', color='g')  # 第 1 个特征
        axes[i, j].plot(sample_data[:, 2], label='Feature 2', color='b')  # 第 2 个特征

        # 设置图例和标题
        axes[i, j].set_title(f'Group {key}, Sample {j + 1}')
        axes[i, j].grid()

plt.tight_layout()
plt.show()
