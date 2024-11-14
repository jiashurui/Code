import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datareader.datareader_stu_1111 import simple_get_stu_1111_all_features, get_features
from prototype.constant import Constant

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


# 计算不同簇数的 SSE
sse = []
k_values = range(1, 12)
for k in k_values:
    # random_state: Cluster中心の初期化を固定する
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(normal_result)
    sse.append(kmeans.inertia_)  # inertia_ 是簇内平方误差和

# Elbow Method to find best K for Kmeans
plt.plot(k_values, sse, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()