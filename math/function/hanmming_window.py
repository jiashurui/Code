import numpy as np
import matplotlib.pyplot as plt

# 定义窗口长度
N = 500

# 生成 Hamming 窗
hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

# 绘制 Hamming 窗
plt.figure(figsize=(10, 4))
plt.plot(hamming_window, label='Hamming Window', color='orange')
plt.title('Hamming Window')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()