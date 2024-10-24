import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming
from scipy.stats import kurtosis, skew
from scipy.spatial.transform import Rotation as R

from prototype.constant import Constant
from prototype.global_tramsform import transform_sensor_data_to_df, transform_sensor_data_to_np
from utils.slidewindow import slide_window2

all_save_base_path = './child/202303/all/'
individual = './child/202303/individual/'

# 统计所有儿童整体行走数据
def show_child_hist_stat():
    big_df = read_data()

    # 创建 1 行 3 列的子图，图像大小为 18x5
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('child all data Histogram', fontsize=20)

    # 绘制每个子图的直方图
    data_list = [big_df['accx'], big_df['accy'], big_df['accz']]
    titles = ['X', 'Y', 'Z']
    col_name = ['accx', 'accy', 'accz']

    for i, ax in enumerate(axes):
        ax.hist(data_list[i], bins=30, color='blue', edgecolor='black', alpha=0.7)
        title = f'2023_03 Child acceleration {titles[i]}'
        ax.set_title(title)
        ax.set_xlabel('Value(m/s2)')  # 设置每个子图的 X 轴标签
        ax.set_ylabel('Number')  # 设置每个子图的 Y 轴标签

    # 保存整个图形
    fig.savefig(f'{all_save_base_path}Child_all_data_histogram.png', dpi=300)

    # 使用 groupby 根据 'Category' 列分组
    groups = list(big_df.groupby('X'))

    for group in groups:
        label = Constant.ChildWalk.action_map_en.get(group[0])
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
    groups = list(data_transformed.groupby('X'))

    for group in groups:
        label = Constant.ChildWalk.action_map_en.get(group[0])
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
def show_child_hist_stat2():
    big_df = read_data()
    big_df = big_df.iloc[:, 1:10]

    # 计算特征量
    df_stat, df_pearson = calc_df_features(big_df)

    # 计算平均fft
    fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq = calc_df_avg_fft(big_df)
    save_fft_result(fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq, f'{all_save_base_path}fft_all_avg_result.png')
    df_stat.to_csv(f'{all_save_base_path}Child_all_data_features.csv')
    df_pearson.to_csv(f'{all_save_base_path}Child_all_data_pearson.csv')

# 展示2023年所有的儿童的步行特征量(按照标签进行分组展示)
def show_child_hist_stat3():
    big_df = read_data()
    # 使用 group by 根据 'Category' 列分组
    groups = list(big_df.groupby('X'))

    for group in groups:
        label = Constant.ChildWalk.action_map_en.get(group[0])
        data = group[1]
        data = data.iloc[:, 1:10]

        # 计算平均fft
        fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq = calc_df_avg_fft(data)
        save_fft_result(fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq,
                        f'{all_save_base_path}fft_{label}_avg_result.png')

        # 计算特征量
        df_stat, df_pearson = calc_df_features(data)
        df_stat.to_csv(f'{all_save_base_path}Child_{label}_features.csv')
        df_pearson.to_csv(f'{all_save_base_path}Child_{label}_pearson.csv')

# 每个个体之间的差异
def show_child_hist_stat4():
    df = read_data()

    # 使用 group by 根据 'Category' 列分组
    groups = list(df.groupby('object'))

    all_individual = []
    for group in groups:
        label = group[0]
        data = group[1]
        data = data.iloc[:, 1:10]

        # 计算平均fft
        fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq = calc_df_avg_fft(data)
        save_fft_result(fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq,
                        f'{individual}fft_{label}_avg_result.png')

        df_stat, _ = calc_df_features(data)
        all_individual.append(df_stat)
    all_individual_stat_data = np.array(all_individual)
    std_values = np.std(all_individual_stat_data, axis=0)
    std_df = pd.DataFrame(std_values, index=all_individual[0].index , columns=all_individual[0].columns)
    std_df.to_csv(f'{individual}Child_individual_features.csv')

# 读取数据
def read_data(file_name = '../data/child/2023_03/merged_data/*.csv'):
    file_list = glob.glob(file_name)
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    return big_df

# 计算数据的各维度特征量
def calc_df_features(df):
    #
    df_mean = df.mean()
    df_min = df.min()
    df_max = df.max()
    df_median = df.median()
    df_std = df.std()
    # 计算变异系数 (CV)
    cv = df.std() / df.mean()
    # 偏度(Skewness)
    skewness = df.apply(lambda x: skew(x))
    # 峰度(Kurtosis)
    kurt = df.apply(lambda x: kurtosis(x))
    # 信号功率 (Signal Power)
    signal_power = df.apply(lambda x: np.mean(x ** 2))
    # 二乘平方根 (Root Mean Square, RMS)
    rms = df.apply(lambda x: np.sqrt(np.mean(x ** 2)))
    # 皮尔森相关系数
    df_pearson = df.corr(method='pearson')

    df_stat = pd.concat([df_mean, df_min, df_max, df_median, df_std, cv, skewness, kurt, signal_power, rms], axis=1)
    df_stat.columns = ['mean', 'min', 'max', 'median', 'std', 'coefficient variation', 'skewness', 'kurt',
                       'signal_power', 'rms']

    return df_stat, df_pearson

# 对单个dataframe整体进行FFT变换
def calc_df_fft(df):
    T = 0.1  # 采样周期为 0.1 秒（10Hz)
    df_acc_x = df['accx'].values
    df_acc_y = df['accy'].values
    df_acc_z = df['accz'].values

    N = len(df_acc_x)

    # 前处理: 去除直流成分
    df_acc_x = df_acc_x - np.mean(df_acc_x)
    df_acc_y = df_acc_y - np.mean(df_acc_y)
    df_acc_z = df_acc_z - np.mean(df_acc_z)

    # 计算傅里叶变换 (对原始数据进行 汉明窗变换)
    df_acc_x = calc_hanmming_window(df_acc_x, N)
    df_acc_y = calc_hanmming_window(df_acc_y, N)
    df_acc_z = calc_hanmming_window(df_acc_z, N)

    # FFT
    fft_acc_x = fft(df_acc_x)
    fft_acc_y = fft(df_acc_y)
    fft_acc_z = fft(df_acc_z)

    # 计算频率
    freq_acc_x = fftfreq(N, T)[:N // 2]

    fft_x_result_scaling = 2.0 / N * np.abs(fft_acc_x[:N // 2])
    fft_y_result_scaling = 2.0 / N * np.abs(fft_acc_y[:N // 2])
    fft_z_result_scaling = 2.0 / N * np.abs(fft_acc_z[:N // 2])

    # 绘制傅里叶变换结果
    # fig, axs = plt.subplots(3, 1, figsize=[10, 5])
    #
    # axs[0].plot(freq_acc_x, fft_x_result_scaling, c='r')
    # axs[1].plot(freq_acc_x, fft_y_result_scaling, c='g')
    # axs[2].plot(freq_acc_x, fft_z_result_scaling, c='b')
    # axs[0].set_title('FFT AccX')
    # axs[1].set_title('FFT AccY')
    # axs[2].set_title('FFT AccZ')
    # fig.tight_layout()
    # plt.show()
    return fft_x_result_scaling, fft_y_result_scaling, fft_z_result_scaling, freq_acc_x

# 对一个dataframe分段计算FFT,然后合并平均FFT的结果
def calc_df_avg_fft(df):
    list_windows = slide_window2(df, 100, 0.5)
    fft_x_list = []
    fft_y_list = []
    fft_z_list = []
    freq = 0.0
    for data_window in list_windows:
        fft_x, fft_y, fft_z, freq_x = calc_df_fft(data_window)
        fft_x_list.append(fft_x)
        fft_y_list.append(fft_y)
        fft_z_list.append(fft_z)

        # 所有FFT结果一样
        freq = freq_x

    # fft_x_avg_seria
    fft_x_avg_series = np.array(fft_x_list).mean(axis=0)
    fft_y_avg_series = np.array(fft_y_list).mean(axis=0)
    fft_z_avg_series = np.array(fft_z_list).mean(axis=0)

    return fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq

# 保存FFT变换的结果
def save_fft_result(fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq , file_name):
    # 绘制傅里叶变换结果
    fig, axs = plt.subplots(3, 1, figsize=[10, 5])
    axs[0].plot(freq, fft_x_avg_series, c='r')
    axs[1].plot(freq, fft_y_avg_series, c='g')
    axs[2].plot(freq, fft_z_avg_series, c='b')
    axs[0].set_title('FFT AccX')
    axs[1].set_title('FFT AccY')
    axs[2].set_title('FFT AccZ')
    fig.tight_layout()
    plt.savefig(f'{file_name}', dpi=300)

# 对数据进行汉明窗变换
def calc_hanmming_window(data, N):
    hamming_window = hamming(N)
    return hamming_window * data

# 展示儿童进行全局变换后的结果(使用我自己的算法)
def show_child_after_transformed():
    df = read_data('../data/child/2023_03/merged_data/A-1_DSCF0001.csv')
    df = df.iloc[:, 1:]
    data_transformed,rpy = transform_sensor_data_to_np(df.values)

    degrees_array = np.degrees(rpy.astype(float))

    df_rpy = pd.DataFrame(degrees_array, columns=['my_roll','my_pitch','my_yaw'])
    df_transformed = pd.DataFrame(data_transformed, columns=df.columns)

    data = pd.concat([df_transformed,df_rpy], axis=1)

    # 设置长方体的初始顶点坐标
    def get_cube_vertices():
        return np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ])

    # 获取长方体的面，用于绘制
    def get_faces(vertices):
        return [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]

    # 创建画布和 3D 轴
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置轴的范围
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # 初始化长方体
    vertices = get_cube_vertices()

    # 更新长方体的位置和旋转
    def update(frame):
        ax.clear()  # 清除之前的绘制

        # 设置轴的范围和标签（每次清除之后需要重新设置）
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # 获取当前的 roll, pitch, yaw
        roll, pitch, yaw = degrees_array[frame]

        # 使用 scipy.spatial.transform.Rotation 进行旋转
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        rotated_vertices = rotation.apply(vertices)

        # 绘制长方体的面，使用不同的颜色表示不同的面
        faces = get_faces(rotated_vertices)
        face_colors = ['blue', 'white', 'green', 'white', 'red', 'white']
        for face, color in zip(faces, face_colors):
            if color:
                cube = Poly3DCollection([face], alpha=0.5, edgecolor='k', facecolor=color)
            else:
                cube = Poly3DCollection([face], alpha=0.0, edgecolor='k')
            ax.add_collection3d(cube)

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(degrees_array), interval=1000)

    # 显示动画
    plt.show()

    # # 创建二维散点图
    # show_size = 20
    # data = data.iloc[:show_size, :]
    # df = df.iloc[:show_size, :]
    # plt.figure(figsize=(10, 6))
    # plt.plot(data.index, data['accx'], color='r', linestyle='-',label='accx')
    # plt.plot(data.index, data['accy'], color='g', linestyle='-',label='accy')
    # plt.plot(data.index, data['accz'], color='b', linestyle='-',label='accz')
    #
    # plt.plot(df.index, df['accx'], color='#8B0000', linestyle='--', label='accx_b')
    # plt.plot(df.index, df['accy'], color='#006400', linestyle='--', label='accy_b')
    # plt.plot(df.index, df['accz'], color='#00008B', linestyle='--', label='accz_b')
    #
    # for i in range(len(data)):
    #     label_text = f"({data['my_roll'][i]:.1f},{data['my_pitch'][i]:.1f},{data['my_yaw'][i]:.1f})"
    #     plt.text(data.index[i], data['accx'][i], label_text, fontsize=4, color='black', ha='center', va='bottom')
    #
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    show_child_hist_stat4()
