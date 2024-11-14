import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datareader.datareader_stu_1111 import simple_get_stu_1111_all_features, get_features
from prototype.constant import Constant

K = 4
features_number = 9
slice_length = 40
# 全局变换之后的大学生数据(全局变换按照frame进行)
origin_data = simple_get_stu_1111_all_features(slice_length, type= 'df', with_rpy= True)
origin_data_np = np.array(origin_data)

features_list, features_name = get_features(origin_data)

train_data = np.array(features_list)
# 必须要用PCA降低维度, 不然90维度Kmeans 结果很糟糕,几乎没法分辨
pca = PCA(n_components=0.70, random_state=3407)
# ica = FastICA(n_components=10, random_state=3407)

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
# plt.scatter(normal_result[:, 0], normal_result[:, 1],color='lightblue', alpha=0.5, s=5)  # 淡蓝色, 半透明, 点大小为1
# plt.show()

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=K, random_state=123)
kmeans.fit(normal_result)
# 获取聚类结果
labels = kmeans.labels_  # 每个样本的聚类标签
centroids = kmeans.cluster_centers_  # 聚类质心



print(f'Simples number:{train_data.shape[0]}')

for i in range(K):
    indices = np.where(labels == i)[0]

    all_gt = [0] * 12
    print(f'Cluster {i}: {len(indices)}')

    for j in range(len(indices)):
        true_label = int(origin_data_np[indices[j], 1, 9].item())
        all_gt[true_label] += 1

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

        ground_truth_name = Constant.uStudent_1111.action_map_en_reverse.get(ground_truth)

        axs[i, j].set_title(f'Cluster: {i}, Data No: {index[j]}, Ground Truth:{ground_truth_name}')
        axs[i, j].set_xlabel('time')
        axs[i, j].set_ylabel('value')

        axs[i, j].plot(num[:, 0], label='acc X')
        axs[i, j].plot(num[:, 1], label='acc Y')
        axs[i, j].plot(num[:, 2], label='acc Z')

plt.tight_layout()  # 自动调整子图之间的间距
plt.show()