import torch
import matplotlib.pyplot as plt

# 创建一个一维的时间序列
t = torch.linspace(0, 2 * torch.pi, 100)
signal = torch.sin(t * 3)  # 频率为3的正弦波


plt.plot(signal)
# 进行快速傅里叶变换
fft_result = torch.fft.fft(signal)

plt.plot(fft_result)
plt.show()
# 查看结果
print(fft_result)
