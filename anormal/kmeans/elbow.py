# 大学生
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from prototype.student.datareader_stu import get_stu_all_features

train_data = get_stu_all_features(20)[:, :, 0: 9]
# 9 features
train_data = train_data.reshape(-1, 9 * 20)
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