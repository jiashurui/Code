# 大学生
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datareader.datareader_stu import get_stu_all_features, simple_get_stu_all_features
from statistic.stat_common import calc_df_features

origin_data = simple_get_stu_all_features(20, type= 'df')
origin_data_np = np.array(origin_data)

features_list = []
for d in origin_data:
    df_features, _ = calc_df_features(d.iloc[:, :9])
    features_list.append(df_features.values.flatten())

train_data = np.array(features_list)
pca = PCA(n_components=5, random_state=3407)
scaler = StandardScaler()
# n_components specifies how many principal components to keep
normal_latent = scaler.fit_transform(train_data)
normal_result = pca.fit_transform(normal_latent)

# 计算不同簇数的 SSE
sse = []
k_values = range(1, 10)
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