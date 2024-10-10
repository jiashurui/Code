import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
t_sne = TSNE(n_components=2, random_state=3407)

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
    plt.scatter(normal_result[:, 0], normal_result[:, 1], color='blue')
    plt.scatter(abnormal_result[:, 0], abnormal_result[:, 1], color='red')
    plt.title(title)
    plt.show()

