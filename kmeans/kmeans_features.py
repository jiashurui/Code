import stat

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from datareader.child_datareader import get_child_all_features
from datareader.datareader_stu import get_stu_all_features, simple_get_stu_all_features
from prototype.constant import Constant
from statistic.stat_common import calc_df_features
from utils.dict_utils import find_key_by_value

# K = 6 に設定する
K = 6
features_number = 9
slice_length = 20
# 大学生 - 9 features
# train_data = get_stu_all_features(slice_length)

# 大学生 - 3 features(加速度のみ)

# origin_data = get_stu_all_features(slice_length)

# 全局变换之后的大学生数据(全局变换按照frame进行)
origin_data = simple_get_stu_all_features(slice_length, type= 'df')

features_list = []
for d in origin_data:
    df_features = calc_df_features(d)
    features_list.append(df_features)

train_data = np.array(features_list)

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=K, random_state=123)
kmeans.fit(train_data)


# 获取聚类结果
labels = kmeans.labels_  # 每个样本的聚类标签
centroids = kmeans.cluster_centers_  # 聚类质心

print(f'Simples number:{train_data.shape[0]}')

true_label_in_cluster = [] * 6
for i in range(K):
    indices = np.where(labels == i)[0]

    all_gt = [0] * K
    print(f'Cluster {i}: {len(indices)}')

    for j in range(len(indices)):
        true_label = int(origin_data[indices[j], 1, 9].item())
        all_gt[true_label - 1] += 1

    true_label_in_cluster.append(all_gt)
    print('    ground truth:', all_gt)


# choose random data to show

fig, axs = plt.subplots(6,5, figsize=(30, 30))

for i in range(K):
    # index arr
    index = np.random.choice(np.where(labels == i)[0],size=5, replace=False)

    # len(index) == 5
    for j in range(len(index)):

        num = origin_data[index[j]].numpy()
        # (data_num , seq_data_index , feature(第9个是标签))
        ground_truth = int(origin_data[index[j], 1, 9].item())

        ground_truth_name = find_key_by_value(Constant.uStudent.action_map_en,ground_truth)

        axs[i, j].set_title(f'Cluster: {i}, Data No: {index[j]}, Ground Truth:{ground_truth_name}')
        axs[i, j].set_xlabel('time')
        axs[i, j].set_ylabel('value')

        axs[i, j].plot(num[:, 0], label='acc X')
        axs[i, j].plot(num[:, 1], label='acc Y')
        axs[i, j].plot(num[:, 2], label='acc Z')

plt.tight_layout()  # 自动调整子图之间的间距
plt.show()