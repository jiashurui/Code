import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from prototype.constant import Constant
from prototype.global_tramsform import transform_sensor_data_to_df

save_base_path = './mHealth/'
def show_mh_hist_stat():
    big_df = read_data()

    # forearm data
    big_df = big_df.iloc[:, 14:24]

    # 创建 1 行 3 列的子图，图像大小为 18x5
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('mHealth all data Histogram', fontsize=20)

    # 绘制每个子图的直方图
    data_list = [big_df['arm_x'], big_df['arm_y'], big_df['arm_z']]
    titles = ['X', 'Y', 'Z']
    col_name = ['arm_x', 'arm_y', 'arm_z']

    for i, ax in enumerate(axes):
        ax.hist(data_list[i], bins=30, color='blue', edgecolor='black', alpha=0.7)
        title = f'mHealth arm acceleration {titles[i]}'
        ax.set_title(title)
        ax.set_xlabel('Value(m/s2)')  # 设置每个子图的 X 轴标签
        ax.set_ylabel('Number')  # 设置每个子图的 Y 轴标签

    # 保存整个图形
    fig.savefig(f'{save_base_path}all/mHealth_all_data_histogram.png', dpi=300)

    # 使用 groupby 根据 'Category' 列分组
    groups = list(big_df.groupby('label'))

    for group in groups:
        label = Constant.mHealth.action_map_reverse.get(group[0])
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
        fig_label.savefig(f'{save_base_path}all_group_by_label/pictures/{label}_hist.png', dpi=300)

    # Global Transformed
    data_transformed = transform_sensor_data_to_df(big_df)
    groups = list(data_transformed.groupby('label'))

    for group in groups:
        label = Constant.mHealth.action_map_reverse.get(group[0])
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
        fig_label.savefig(f'{save_base_path}all_group_by_label/pictures/{label}_transformed_hist.png', dpi=300)

# 统计mHealth 所有数据
def stat_mh_all_data():
    df = read_data()
    # forearm data
    df = df.iloc[:, 14:23] # not include label

    df_stat, df_pearson = calc_df_features(df)
    df_stat.to_csv(f'{save_base_path}/all/mHealth_all_features.csv')

def stat_mh_label_data():
    df = read_data()
    # 使用 group by 根据 'Category' 列分组
    groups = list(df.groupby('label'))

    for group in groups:
        label = Constant.mHealth.action_map_reverse.get(group[0])
        data = group[1]
        data = data.iloc[:, list(range(14, 23))]
        df_stat, df_pearson = calc_df_features(data)
        df_stat.to_csv(f'{save_base_path}all_group_by_label/stat/mHealth_{label}_features.csv')
        df_pearson.to_csv(f'{save_base_path}all_group_by_label/stat/mHealth_{label}_pearson.csv')


# 统计mHealth 个人向数据
def stat_mh_indivdual_data():
    df = read_data()
    # 使用 group by 根据 'Category' 列分组
    groups = list(df.groupby('object'))
    all_individual = []

    for group in groups:
        label = group[0]
        data = group[1]
        data = data.iloc[:, list(range(14, 23))]
        df_stat, _ = calc_df_features(data)
        all_individual.append(df_stat)

    all_individual_stat_data = np.array(all_individual)
    std_values = np.std(all_individual_stat_data, axis=0)
    std_df = pd.DataFrame(std_values, index=all_individual[0].index, columns=all_individual[0].columns)
    std_df.to_csv(f'{save_base_path}individual/Child_individual_features.csv')


def read_data():
    file_list = glob.glob('../data/mHealth/mHealth_*.log')
    appended_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name, sep='\t', header=None)
        data.columns = Constant.mHealth.data_columns
        object = re.findall(r'\d+', file_name)[0]
        # 人
        data['object'] = object
        appended_data.append(data)

    big_df = pd.concat(appended_data, ignore_index=True)
    return big_df


def calc_df_features(df):
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
    stat_mh_indivdual_data()
