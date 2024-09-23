import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# 假设我们有数据 X
X = np.random.randn(10000, 22)

# 计算不同簇数的 SSE
sse = []
k_values = range(1, 300)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)  # inertia_ 是簇内平方误差和

# 绘制肘部法图
plt.plot(k_values, sse, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()
