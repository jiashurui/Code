import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 假设有 1000 组数据，每组有 100 个时间点，每个时间点有 10 维特征
data = np.random.rand(1000, 100, 10)

# 1. 标准化数据
scaler = TimeSeriesScalerMeanVariance()
data_scaled = scaler.fit_transform(data)

# 2. 使用 KMeans 进行聚类（DTW距离）
n_clusters = 5
kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, random_state=42)
y_pred = kmeans.fit_predict(data_scaled)

# 3. 从每个簇中随机选择几组样本进行可视化
n_samples_per_cluster = 5  # 每个簇可视化 5 个样本
plt.figure(figsize=(12, 10))

for i in range(n_clusters):
    cluster_indices = np.where(y_pred == i)[0]
    selected_indices = np.random.choice(cluster_indices, n_samples_per_cluster, replace=False)

    plt.subplot(n_clusters, 1, i + 1)
    for idx in selected_indices:
        plt.plot(data_scaled[idx].ravel(), alpha=0.6)
    plt.title(f"Cluster {i + 1} Sample Time Series")
    plt.xlabel("Time Step")
    plt.ylabel("Feature Value")

plt.tight_layout()
plt.show()
