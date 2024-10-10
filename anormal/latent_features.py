import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 假设time_series_data是你的时间序列数据 (例如：shape = (num_samples, time_steps, num_features))
# 将其划分为固定长度的窗口，假设窗口长度为n_window
def create_time_windows(time_series_data, window_size):
    n_samples = len(time_series_data) - window_size + 1
    windows = np.array([time_series_data[i:i+window_size] for i in range(n_samples)])
    return windows.reshape(n_samples, -1)  # 将窗口展平为二维矩阵

# 生成时间窗口特征
time_series_data = np.random.randn(1000)  # 假设有1000个时间点的一维时间序列
window_size = 10  # 每个窗口的长度
windows = create_time_windows(time_series_data, window_size)

# 标准化数据
scaler = StandardScaler()
windows_scaled = scaler.fit_transform(windows)

# 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
windows_tsne = tsne.fit_transform(windows_scaled)

# 可视化降维结果
plt.scatter(windows_tsne[:, 0], windows_tsne[:, 1])
plt.title('t-SNE on Time Series Data')
plt.show()
