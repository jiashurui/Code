import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles
from sklearn.metrics import pairwise_distances
import numpy as np
# plt.figure(figsize=(8, 8))
# plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)
from collections import Counter

# plt.subplot(321)

count = 100
K = 5

plt.title("One informative feature, one cluster per class")
X1, Y1 = make_classification(
    n_samples=count,
    n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1,
    scale=10.0
)
# 定义颜色映射
colors = plt.cm.tab10

plot = plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, cmap=colors, s=10, )

data = np.array([0., 0.])

# plt.show()
distance_array = np.empty([100, 3])

for i in range(count):
    # pairwise_distances(X1[i], data, metric='euclidean', n_jobs=-1)
    distance = np.linalg.norm(data - X1[i])
    distance_array[i, 0] = X1[i][0]
    distance_array[i, 1] = X1[i][1]
    distance_array[i, 2] = distance

min_indices = np.argsort(distance_array[:, 2])[:K]

# connect
count_true = np.int8(0)
count_false = np.int8(0)

for index in min_indices:
    if Y1[index] == 1:
        count_true += 1
    else:
        count_false += 1

result = -1
if count_true > count_false:
    result = np.array([3])
else:
    result = np.array([4])
print(result)
plt.scatter(data[0], data[1], c=result, marker='o', cmap=colors, s=20, )

plt.plot([data[0], distance_array[min_indices][0][0]], [data[1], distance_array[min_indices][0][1]], color='red',
         linestyle='solid')
plt.plot([data[0], distance_array[min_indices][1][0]], [data[1], distance_array[min_indices][1][1]], color='red',
         linestyle='solid')
plt.plot([data[0], distance_array[min_indices][2][0]], [data[1], distance_array[min_indices][2][1]], color='red',
         linestyle='solid')
plt.plot([data[0], distance_array[min_indices][3][0]], [data[1], distance_array[min_indices][3][1]], color='red',
         linestyle='solid')
plt.plot([data[0], distance_array[min_indices][4][0]], [data[1], distance_array[min_indices][4][1]], color='red',
         linestyle='solid')

plt.show()
