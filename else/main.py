import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# 生成三维示例数据
X = np.random.rand(300, 3)  # 300个样本，每个样本有三个特征

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.labels_  # 聚类标签
centroids = kmeans.cluster_centers_  # 聚类质心

# 使用 Plotly 进行三维动态可视化
fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels.astype(str),
                    title="K-Means Clustering (3D)",
                    labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3'})

# 添加聚类质心
fig.add_scatter3d(x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
                  mode='markers', marker=dict(size=10, color='red'), name='Centroids')

# 显示图形
fig.show()
