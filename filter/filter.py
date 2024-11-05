import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np

# 低通滤波器设计函数
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 低通滤波函数
def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
