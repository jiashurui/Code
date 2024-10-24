import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial.transform import Rotation as R

from prototype.constant import Constant
from prototype.global_tramsform import transform_sensor_data_to_df, transform_sensor_data_to_np
from stat_common import calc_fft_spectral_energy, spectral_entropy, calc_hanmming_window, calc_df_features, calc_df_fft, \
    save_fft_result

from utils.slidewindow import slide_window2

all_save_base_path = './realworld/all/'
individual = './realworld/individual/'


# 统计realworld整体行走数据
def show_realworld_hist_stat():
    big_df = read_data()

    # 创建 1 行 3 列的子图，图像大小为 18x5
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('realworld all data Histogram', fontsize=20)

    # 绘制每个子图的直方图
    data_list = [big_df['acc_attr_x'], big_df['acc_attr_y'], big_df['acc_attr_z']]
    titles = ['X', 'Y', 'Z']
    col_name = ['acc_attr_x', 'acc_attr_y', 'acc_attr_z']

    for i, ax in enumerate(axes):
        ax.hist(data_list[i], bins=30, color='blue', edgecolor='black', alpha=0.7)
        title = f'Realworld acceleration {titles[i]}'
        ax.set_title(title)
        ax.set_xlabel('Value(m/s2)')  # 设置每个子图的 X 轴标签
        ax.set_ylabel('Number')  # 设置每个子图的 Y 轴标签

    # 保存整个图形
    fig.savefig(f'{all_save_base_path}Realworld_histogram.png', dpi=300)

    # 使用 groupby 根据 'Category' 列分组
    groups = list(big_df.groupby('label'))

    for group in groups:
        label = Constant.RealWorld.action_map_reverse.get(group[0])
        data = group[1]
        fig_label, axes_label = plt.subplots(1, 3, figsize=(12, 6))
        fig_label.suptitle(f'{label}_hist', fontsize=20)

        for i, ax in enumerate(axes_label):
            ax.hist(data[col_name[i]], bins=30, color='blue', edgecolor='black', alpha=0.7)
            title = f'{titles[i % 3]}, label: {label}'
            ax.set_title(title)
            ax.set_xlabel('Value(m/s2)')
            ax.set_ylabel('Number')

        # 保存当前子图的文件，使用 fig_label.suptitle 的标题作为文件名
        fig_label.savefig(f'{all_save_base_path}{label}_hist.png', dpi=300)

    # Global Transformed
    data_transformed = transform_sensor_data_to_df(big_df)
    groups = list(data_transformed.groupby('label'))

    for group in groups:
        label = Constant.RealWorld.action_map_reverse.get(group[0])
        data = group[1]
        fig_label, axes_label = plt.subplots(1, 3, figsize=(12, 6))
        fig_label.suptitle(f'{label}_transformed_hist', fontsize=20)

        for i, ax in enumerate(axes_label):
            ax.hist(data[col_name[i]], bins=30, color='blue', edgecolor='black', alpha=0.7)
            title = f'{titles[i % 3]}, label: {label}'
            ax.set_title(title)
            ax.set_xlabel('Value(m/s2)')
            ax.set_ylabel('Number')

        # 保存当前子图的文件，使用 fig_label.suptitle 的标题作为文件名
        fig_label.savefig(f'{all_save_base_path}{label}_transformed_hist.png', dpi=300)


# 特徴量：平均值,最小值,最大值,中央值,
# 标准差, 变异系数(CV),偏度(Skewness),"Kurtosis"（峰度）
# 信号power, 二乘平方根, peak强度, 皮尔逊相关系数
def show_realworld_hist_stat2():
    big_df = read_data()

    # 计算特征量
    df_stat, df_pearson = calc_df_features(big_df)

    # 计算平均fft
    fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq, df_freq_stat = calc_df_avg_fft(big_df)
    save_fft_result(fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq,
                    f'{all_save_base_path}fft_avg_result.png')
    df_stat.to_csv(f'{all_save_base_path}Realworld_features.csv')
    df_freq_stat.to_csv(f'{all_save_base_path}Realworld_freq_features.csv')

    df_pearson.to_csv(f'{all_save_base_path}Realworld_all_data_pearson.csv')


# 展示realworld的特征量(按照标签进行分组展示)
def show_realworld_hist_stat3():
    big_df = read_data()
    # 使用 group by 根据 'Category' 列分组
    groups = list(big_df.groupby('label'))

    for group in groups:
        label = Constant.RealWorld.action_map_reverse.get(group[0])
        data = group[1]

        # 计算平均fft
        fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq, df_freq_stat = calc_df_avg_fft(data)
        save_fft_result(fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq,
                        f'{all_save_base_path}fft_{label}_avg_result.png')

        # 计算特征量
        df_stat, df_pearson = calc_df_features(data)
        df_stat.to_csv(f'{all_save_base_path}Realworld_{label}_features.csv')
        df_freq_stat.to_csv(f'{all_save_base_path}Realworld_{label}_freq_features.csv')

        df_pearson.to_csv(f'{all_save_base_path}Realworld_{label}_pearson.csv')


# 读取数据
def read_data(file_name='../data/realworld/*/chest_merged.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    return big_df

# 对一个dataframe分段计算FFT,然后合并平均FFT的结果
def calc_df_avg_fft(df):
    list_windows = slide_window2(df, 150, 0.5)
    fft_x_list = []
    fft_y_list = []
    fft_z_list = []
    freq = 0.0

    features_list = []
    for data_window in list_windows:
        fft_x, fft_y, fft_z, freq_x, max_freq_x, max_freq_y, max_freq_z = calc_df_fft(data_window, acc_x_name='acc_attr_x', acc_y_name='acc_attr_y', acc_z_name='acc_attr_z', T=0.02)
        x_spec_energy, y_spec_energy, z_spec_energy, spec_total = calc_fft_spectral_energy(data_window, acc_x_name='acc_attr_x', acc_y_name='acc_attr_y', acc_z_name='acc_attr_z', T=50)
        x_spec_entropy, y_spec_entropy, z_spec_entropy, entropy_total = spectral_entropy(data_window, acc_x_name='acc_attr_x', acc_y_name='acc_attr_y', acc_z_name='acc_attr_z', T=50)

        fft_x_list.append(fft_x)
        fft_y_list.append(fft_y)
        fft_z_list.append(fft_z)

        # 所有FFT结果一样
        freq = freq_x
        # 特征值(能量与能量熵)
        features_list.append((x_spec_energy, y_spec_energy, z_spec_energy, spec_total,
                              x_spec_entropy, y_spec_entropy, z_spec_entropy, entropy_total,
                              max_freq_x, max_freq_y, max_freq_z))

    feature_arr = np.array(features_list)

    # 所有数据,每个频段上的平均(1hz, 2hz, 3hz)
    fft_x_avg_series = np.array(fft_x_list).mean(axis=0)
    fft_y_avg_series = np.array(fft_y_list).mean(axis=0)
    fft_z_avg_series = np.array(fft_z_list).mean(axis=0)

    # 所有数据,全频段上的能量和,的平均energy(1hz+2hz+3hz)/avg
    energy_x_avg_series = np.mean(feature_arr[:, 0])
    energy_y_avg_series = np.mean(feature_arr[:, 1])
    energy_z_avg_series = np.mean(feature_arr[:, 2])
    energy_t_avg_series = np.mean(feature_arr[:, 3])

    # 所有数据,全频段上的能量熵,的平均entropy(1hz+2hz+3hz)/avg
    entropy_x_avg_series = np.mean(feature_arr[:, 4])
    entropy_y_avg_series = np.mean(feature_arr[:, 5])
    entropy_z_avg_series = np.mean(feature_arr[:, 6])
    entropy_t_avg_series = np.mean(feature_arr[:, 7])

    # 频域最大值(hz)
    freq_max_x = np.mean(feature_arr[:, 8])
    freq_max_y = np.mean(feature_arr[:, 9])
    freq_max_z = np.mean(feature_arr[:, 10])

    # 保存特征分析结果
    df_stat = pd.DataFrame([energy_x_avg_series, energy_y_avg_series, energy_z_avg_series, energy_t_avg_series, \
                            entropy_x_avg_series, entropy_y_avg_series, entropy_z_avg_series, entropy_t_avg_series,
                            freq_max_x, freq_max_y, freq_max_z])
    df_stat.index = ['energy_x', 'energy_y', 'energy_z', 'energy_total',
                     'entropy_x', 'entropy_y', 'entropy_z', 'entropy_total',
                     'freq_max_x', 'freq_max_y', 'freq_max_z']
    df_stat.columns = ['value']

    return fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq, df_stat

if __name__ == '__main__':
    show_realworld_hist_stat3()
