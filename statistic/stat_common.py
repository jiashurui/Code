# 计算一个时间步上, 频谱能量和
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.signal.windows import hamming
from scipy.stats import kurtosis, skew


# 对单个dataframe整体进行FFT变换
def calc_df_fft(df, acc_x_name='accx', acc_y_name='accy', acc_z_name='accz', T=0.1):
    df_acc_x = df[acc_x_name].values
    df_acc_y = df[acc_y_name].values
    df_acc_z = df[acc_z_name].values

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


# 计算一个时间步上, 频谱能量
def calc_fft_spectral_energy(df, acc_x_name='accx', acc_y_name='accy', acc_z_name='accz', T=10):
    df_acc_x = df[acc_x_name].values
    df_acc_y = df[acc_y_name].values
    df_acc_z = df[acc_z_name].values

    # 使用 Welch 方法计算功率谱密度（PSD）
    freqs_x, psd_x = welch(df_acc_x, fs=T, nperseg=len(df))
    freqs_y, psd_y = welch(df_acc_y, fs=T, nperseg=len(df))
    freqs_z, psd_z = welch(df_acc_z, fs=T, nperseg=len(df))

    # 找到指定频率范围 [0, 5Hz] 的索引
    freq_indices = np.where((freqs_x >= 0) & (freqs_x <= 5))[0]

    # 计算频带 [0, 5Hz] 的能量
    spectral_energy_x = np.sum(psd_x[freq_indices])
    spectral_energy_y = np.sum(psd_y[freq_indices])
    spectral_energy_z = np.sum(psd_z[freq_indices])

    # 归一化：除以频率区间内的频率数量 (x + y + 1)
    normalized_energy_x = spectral_energy_x / (5 + 1)
    normalized_energy_y = spectral_energy_y / (5 + 1)
    normalized_energy_z = spectral_energy_z / (5 + 1)

    # 总的频谱能量
    total_spectral_energy = normalized_energy_x + normalized_energy_y + normalized_energy_z

    return normalized_energy_x, normalized_energy_y, normalized_energy_z, total_spectral_energy


# 计算一个时间步上, 频谱熵
def spectral_entropy(df, acc_x_name='accx', acc_y_name='accy', acc_z_name='accz', T=10):
    df_acc_x = df[acc_x_name].values
    df_acc_y = df[acc_y_name].values
    df_acc_z = df[acc_z_name].values

    # 使用Welch方法计算功率谱密度 (PSD)
    _, psd_x = welch(df_acc_x, fs=T, nperseg=len(df))  # 10hz
    _, psd_y = welch(df_acc_y, fs=T, nperseg=len(df))  # 10hz
    _, psd_z = welch(df_acc_z, fs=T, nperseg=len(df))  # 10hz

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


# 定义计算频谱质心频率的函数
# {Param: series: np }
def spectral_centroid(series, sampling_rate=10):
    # 计算傅里叶变换
    spectrum = np.fft.fft(series)
    # 获取频谱的幅度
    magnitude = np.abs(spectrum)
    # 获取频率轴
    frequency = np.fft.fftfreq(len(series), d=1 / sampling_rate)

    # 计算频谱质心
    centroid = np.sum(frequency * magnitude) / np.sum(magnitude)
    return centroid

# 定义计算主频率的函数
# {Param: series: np }
def dominant_frequency(series, sampling_rate=10):
    # 计算傅里叶变换
    spectrum = np.fft.fft(series)
    # 获取频谱的幅度
    magnitude = np.abs(spectrum)
    # 获取频率轴
    freqs = np.fft.fftfreq(len(series), d=1 / sampling_rate)

    # 找到幅值最大的频率
    dominant_freq = freqs[np.argmax(magnitude[1:]) + 1]  # 忽略直流分量 (DC 分量)
    return dominant_freq


# 对数据进行汉明窗变换
def calc_hanmming_window(data, N):
    hamming_window = hamming(N)
    return hamming_window * data


# 计算数据的各维度特征量
def calc_df_features(df):
    # 均值(Mean)
    df_mean = df.mean()
    # 最小值(Min)
    df_min = df.min()
    # 最大值(Max)
    df_max = df.max()
    # 中位值(Median)
    df_median = df.median()
    # 标准差(Standard Deviation)
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
    # 峰值振幅 (Peak-to-Peak Amplitude)
    ptp = df.apply(lambda x: np.ptp(x))
    # 过零率(Zero-Crossing Rate)
    zcr = df.apply(lambda x: np.sum(np.diff(np.sign(x)) != 0) / len(x))
    # 过均值率(Mean Crossing Rate)
    mcr = df.apply(lambda x: np.sum(np.diff(np.sign(x - x.mean())) != 0) / len(x))
    # 对数能量(Log energy) 1e-10 是防止对数取0
    log_energy = df.apply(lambda x: np.log(np.sum(x ** 2) + 1e-10))
    # 方差
    var = df.apply(lambda x: x.var(), axis=1)

    # 皮尔森相关系数
    df_pearson = df.corr(method='pearson')

    df_stat = pd.concat([df_mean, df_min, df_max, df_median, df_std, cv, skewness, kurt, signal_power, rms,
                         ptp, zcr, mcr, log_energy, var], axis=1)
    df_stat.columns = ['mean', 'min', 'max', 'median', 'std', 'coefficient variation', 'skewness', 'kurt',
                       'signal_power', 'rms', 'ptp', 'zcr', 'mcr', 'log_energy', 'var']

    return df_stat, df_pearson

# XYZ 合计信号幅值面积
# Signal Magnitude Area
def calc_acc_sma(acc_x , acc_y , acc_z):
    return np.sum((np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z)))/ len(acc_x)


# 保存FFT变换的结果
def save_fft_result(fft_x_avg_series, fft_y_avg_series, fft_z_avg_series, freq, file_name):
    # 绘制傅里叶变换结果
    fig, axs = plt.subplots(3, 1, figsize=[10, 5])
    axs[0].plot(freq, fft_x_avg_series, c='r')
    axs[1].plot(freq, fft_y_avg_series, c='g')
    axs[2].plot(freq, fft_z_avg_series, c='b')
    axs[0].set_title('FFT AccX')
    axs[1].set_title('FFT AccY')
    axs[2].set_title('FFT AccZ')
    # 限制 x 轴最多显示到 5
    axs[0].set_xlim(0, 5)
    axs[1].set_xlim(0, 5)
    axs[2].set_xlim(0, 5)

    fig.tight_layout()
    plt.savefig(f'{file_name}', dpi=300)
