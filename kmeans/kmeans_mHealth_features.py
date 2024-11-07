import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from datareader.mh_datareader import simple_get_mh_all_features
from prototype.constant import Constant
from statistic.stat_common import calc_df_features, spectral_centroid, dominant_frequency, calculate_ar_coefficients, \
    calc_fft_spectral_energy, spectral_entropy, calc_acc_sma
from utils.dict_utils import find_key_by_value

# K = 6 に設定する
K = 12
simpling = 50
features_number = 9
slice_length = 256
# 全局变换之后的大学生数据(全局变换按照frame进行)
origin_data = simple_get_mh_all_features(slice_length, type='df', with_rpy= True)
origin_data_np = np.array(origin_data)

features_list = []
for d in origin_data:
    df_features, _ = calc_df_features(d.iloc[:, :9])

    # 分别对9维数据XYZ求FFT的能量(结果会变坏)
    aex,aey,aez,aet = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='arm_x', acc_y_name='arm_y', acc_z_name='arm_z', T=simpling)
    gex,gey,gez,get = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='gyro_arm_x', acc_y_name='gyro_arm_y', acc_z_name='gyro_arm_z', T=simpling)
    mex,mey,mez,met = calc_fft_spectral_energy(d.iloc[:, :9], acc_x_name='magnetometer_arm_x', acc_y_name='magnetometer_arm_y', acc_z_name='magnetometer_arm_z', T=simpling)
    df_features['fft_spectral_energy'] = [aex,aey,aez,gex,gey,gez,mex,mey,mez]

    # 分别对9维数据XYZ求FFT的能量(结果会变坏)
    aex,aey,aez,aet = spectral_entropy(d.iloc[:, :9], acc_x_name='arm_x', acc_y_name='arm_y', acc_z_name='arm_z', T=simpling)
    gex,gey,gez,get = spectral_entropy(d.iloc[:, :9], acc_x_name='gyro_arm_x', acc_y_name='gyro_arm_y', acc_z_name='gyro_arm_z', T=simpling)
    mex,mey,mez,met = spectral_entropy(d.iloc[:, :9], acc_x_name='magnetometer_arm_x', acc_y_name='magnetometer_arm_y', acc_z_name='magnetometer_arm_z', T=simpling)
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
# 必须要用PCA降低维度, 不然90维度Kmeans 结果很糟糕,几乎没法分辨
pca = PCA(n_components=0.7, random_state=3407)

# PCA 和T-SNE结果差不错,没什么太大区别
# t_sne = TSNE(n_components=2, random_state=3407, perplexity=50, n_jobs=-1, method='exact')

# StandardScaler > MinMaxScaler(-1,1) > MinMaxScaler(0,1)
scaler = StandardScaler()

# PCA之前必须要进行正则化,不然结果也会很糟糕
normal_latent = scaler.fit_transform(train_data)
normal_result = pca.fit_transform(normal_latent)


explained_variance = pca.explained_variance_ratio_
print("PCA 维度:", len(explained_variance))
print("方差解释率:", explained_variance)
print("方差累计解释率:", np.sum(explained_variance))


# PCA 结果可视化
plt.scatter(normal_result[:, 0], normal_result[:, 1],color='lightblue', alpha=0.5, s=5)  # 淡蓝色, 半透明, 点大小为1
plt.show()

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=K, random_state=123)
kmeans.fit(normal_result)

# 获取聚类结果
labels = kmeans.labels_  # 每个样本的聚类标签
db_score = davies_bouldin_score(normal_result, labels)
print(f"Davies-Bouldin Score: {db_score}")

centroids = kmeans.cluster_centers_  # 聚类质心

print(f'Simples number:{train_data.shape[0]}')

true_label_in_cluster = [] * 6
for i in range(K):
    indices = np.where(labels == i)[0]

    all_gt = [0] * K
    print(f'Cluster {i}: {len(indices)}')

    for j in range(len(indices)):
        true_label = int(origin_data_np[indices[j], 1, 9].item())
        all_gt[true_label - 1] += 1

    true_label_in_cluster.append(all_gt)
    print('    ground truth:', all_gt)


# choose random data to show

fig, axs = plt.subplots(K,5, figsize=(30, 30))

for i in range(K):
    # index arr
    index = np.random.choice(np.where(labels == i)[0],size=5, replace=False)

    # len(index) == 5
    for j in range(len(index)):

        num = origin_data_np[index[j]]
        # (data_num , seq_data_index , feature(第9个是标签))
        ground_truth = int(origin_data_np[index[j], 1, 9].item())

        ground_truth_name = find_key_by_value(Constant.mHealth.action_map, ground_truth)

        axs[i, j].set_title(f'Cluster: {i}, Data No: {index[j]}, Ground Truth:{ground_truth_name}')
        axs[i, j].set_xlabel('time')
        axs[i, j].set_ylabel('value')

        axs[i, j].plot(num[:, 0], label='acc X')
        axs[i, j].plot(num[:, 1], label='acc Y')
        axs[i, j].plot(num[:, 2], label='acc Z')

plt.tight_layout()  # 自动调整子图之间的间距
plt.show()