import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from prototype.constant import Constant
from prototype.global_tramsform import transform_sensor_data_to_df

save_base_path = './child/202303/'


def show_child_hist_stat():
    file_list = glob.glob('../data/child/2023_03/merged_data/*.csv')
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

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
    fig.savefig(f'{save_base_path}Child_all_data_histogram.png', dpi=300)

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
        fig_label.savefig(f'{save_base_path}{label}_hist.png', dpi=300)

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
        fig_label.savefig(f'{save_base_path}{label}_transformed_hist.png', dpi=300)


def show_child_hist_stat2():
    file_list = glob.glob('../data/child/2023_03/merged_data/*.csv')
    appended_data = []

    # 特徴量：平均值,最小值,最大值,中央值,
    # 标准差, 变异系数(CV),偏度(Skewness),"Kurtosis"（峰度）
    # 信号power, 二乘平方根, peak强度, 皮尔逊相关系数

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    big_df = big_df.iloc[:, 1:19]

    df_stat, df_pearson = calc_df_features(big_df)
    df_stat.to_csv(f'{save_base_path}Child_all_data_features.csv')
    df_pearson.to_csv(f'{save_base_path}Child_all_data_pearson.csv')


# 展示2023年所有的儿童的步行特征量(按照标签进行分组展示)
def show_child_hist_stat3():
    file_list = glob.glob('../data/child/2023_03/merged_data/*.csv')
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)

    # 使用 group by 根据 'Category' 列分组
    groups = list(big_df.groupby('X'))

    for group in groups:
        label = Constant.ChildWalk.action_map_en.get(group[0])
        data = group[1]

        data = data.iloc[:, 1:19]

        df_stat, df_pearson = calc_df_features(data)
        df_stat.to_csv(f'{save_base_path}Child_{label}_features.csv')
        df_pearson.to_csv(f'{save_base_path}Child_{label}_pearson.csv')

        print(label, data)


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


if __name__ == '__main__':
    show_child_hist_stat3()
