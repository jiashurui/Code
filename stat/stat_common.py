# 计算一个时间步上, 频谱能量和
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal.windows import hamming
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq

def calc_fft_spectral_energy(df):
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

    # 只保留正频率部分（傅里叶变换的前一半结果）
    fft_acc_x = fft_acc_x[:N // 2]
    fft_acc_y = fft_acc_y[:N // 2]
    fft_acc_z = fft_acc_z[:N // 2]

    # spectral_energy 计算频谱能量
    spectral_energy_x = np.sum(np.abs(fft_acc_x) ** 2)
    spectral_energy_y = np.sum(np.abs(fft_acc_y) ** 2)
    spectral_energy_z = np.sum(np.abs(fft_acc_z) ** 2)

    # 使用L1 范式,计算频谱能量
    total_spectral_energy = spectral_energy_x + spectral_energy_y + spectral_energy_z

    return spectral_energy_x, spectral_energy_y, spectral_energy_z, total_spectral_energy


# 计算一个时间步上, 频谱熵
def spectral_entropy(df):
    df_acc_x = df['accx'].values
    df_acc_y = df['accy'].values
    df_acc_z = df['accz'].values

    # 使用Welch方法计算功率谱密度 (PSD)
    _, psd_x = welch(df_acc_x, fs=10, nperseg=len(df))  # 10hz
    _, psd_y = welch(df_acc_y, fs=10, nperseg=len(df))  # 10hz
    _, psd_z = welch(df_acc_z, fs=10, nperseg=len(df))  # 10hz

    # 归一化 PSD 以形成概率分布
    psd_norm_x = psd_x / np.sum(psd_x)
    psd_norm_y = psd_y / np.sum(psd_y)
    psd_norm_z = psd_z / np.sum(psd_z)

    # 计算 Shannon Entropy
    entropy_x = -np.sum(psd_norm_x * np.log2(psd_norm_x + np.finfo(float).eps))
    entropy_y = -np.sum(psd_norm_y * np.log2(psd_norm_y + np.finfo(float).eps))
    entropy_z = -np.sum(psd_norm_z * np.log2(psd_norm_z + np.finfo(float).eps))

    # 使用L1 范式,进行特征合并
    entropy_total = entropy_x + entropy_y + entropy_z

    return entropy_x, entropy_y, entropy_z, entropy_total


# 对数据进行汉明窗变换
def calc_hanmming_window(data, N):
    hamming_window = hamming(N)
    return hamming_window * data

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

    positive_fft_x = np.abs(fft_acc_x[:N // 2])
    positive_fft_y = np.abs(fft_acc_y[:N // 2])
    positive_fft_z = np.abs(fft_acc_z[:N // 2])

    fft_x_result_scaling = 2.0 / N * positive_fft_x
    fft_y_result_scaling = 2.0 / N * positive_fft_y
    fft_z_result_scaling = 2.0 / N * positive_fft_z

    # 获取最大频率的值
    max_freq_x = freq_acc_x[np.argmax(positive_fft_x)]
    max_freq_y = freq_acc_x[np.argmax(positive_fft_y)]
    max_freq_z = freq_acc_x[np.argmax(positive_fft_z)]

    return fft_x_result_scaling, fft_y_result_scaling, fft_z_result_scaling, freq_acc_x, \
        max_freq_x, max_freq_y, max_freq_z
