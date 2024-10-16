import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
t_sne = TSNE(n_components=2, random_state=3407, perplexity=50, n_jobs=-1)

pca = PCA(n_components=2,random_state=3407)  # n_components specifies how many principal components to keep


def plot_tsne(normal_data, abnormal_data, title):
    # Tensor to Numpy
    normal_data_np_array = normal_data.cpu().numpy()
    abnormal_data_np_array = abnormal_data.cpu().numpy()

    # Scaling
    normal_latent = scaler.fit_transform(normal_data_np_array)
    abnormal_latent = scaler.fit_transform(abnormal_data_np_array)

    # t-SNE
    normal_result = t_sne.fit_transform(normal_latent)
    abnormal_result = t_sne.fit_transform(abnormal_latent)

    # Plot the result after dimension reduction
    plt.scatter(normal_result[:, 0], normal_result[:, 1],
                color='lightblue', alpha=0.5, s=1)  # 淡蓝色, 半透明, 点大小为10
    plt.scatter(abnormal_result[:, 0], abnormal_result[:, 1],
                color='lightcoral', alpha=0.5, s=1)  # 淡红色, 半透明, 点大小为10

    plt.title(title)
    plt.show()

def plot_pca(normal_data, abnormal_data, title):
    # Tensor to Numpy
    normal_data_np_array = normal_data.cpu().numpy()
    abnormal_data_np_array = abnormal_data.cpu().numpy()

    # Scaling
    normal_latent = scaler.fit_transform(normal_data_np_array)
    abnormal_latent = scaler.fit_transform(abnormal_data_np_array)

    # PCA
    normal_result = pca.fit_transform(normal_latent)
    abnormal_result = pca.fit_transform(abnormal_latent)

    # Plot the result after dimension reduction
    plt.scatter(normal_result[:, 0], normal_result[:, 1],
                color='lightblue', alpha=0.5, s=1)  # 淡蓝色, 半透明, 点大小为1
    plt.scatter(abnormal_result[:, 0], abnormal_result[:, 1],
                color='lightcoral', alpha=0.5, s=1)  # 淡红色, 半透明, 点大小为1
    plt.title(title)
    plt.show()

# 对于任意数据, 数据标签, 标签表示含义,进行PCA可视化展示
def plot_data_pca(data , label_list , label_map):

    data = scaler.fit_transform(data)
    result = pca.fit_transform(data)
    plt.figure(figsize=(10, 6))

    # Plot the result after dimension reduction
    plt.scatter(result[:, 0], result[:, 1],
                color='lightblue', alpha=0.5, s=1)  # 淡蓝色, 半透明, 点大小为1

    # 绘制PCA降维后的数据
    for label in np.unique(label_list):
        plt.scatter(result[label_list == label, 0], result[label_list == label, 1], s=3, label=label_map.get(label), alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()

def plot_data_tSNE(data , label_list , label_map):
    data = scaler.fit_transform(data)
    result = t_sne.fit_transform(data)
    plt.figure(figsize=(10, 6))

    # Plot the result after dimension reduction
    plt.scatter(result[:, 0], result[:, 1],
                color='lightblue', alpha=0.5, s=1)  # 淡蓝色, 半透明, 点大小为1

    # 绘制PCA降维后的数据
    for label in np.unique(label_list):
        plt.scatter(result[label_list == label, 0], result[label_list == label, 1], s=3, label=label_map.get(label), alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()