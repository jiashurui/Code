# 小学生
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from datareader.child_datareader import simple_get_child_2024_all_features

window_size = 20
feature_nums = 9


train_data = simple_get_child_2024_all_features(window_size)
train_data = train_data.reshape(-1, feature_nums * window_size)
# 计算不同簇数的 SSE
sse = []
k_values = range(1, 10)
for k in k_values:
    # random_state: Cluster中心の初期化を固定する
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(train_data)
    sse.append(kmeans.inertia_)  # inertia_ 是簇内平方误差和

# Elbow Method to find best K for Kmeans
plt.plot(k_values, sse, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()