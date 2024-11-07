import torch
from matplotlib import pyplot as plt

from gan.lstm_gan import Generator, Discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

slice_length = 256
z_dim = 10       # 噪声向量的维度
hidden_dim = 30  # LSTM 的隐藏层维度
output_dim = 9   # 时间序列的特征维度，即加速度和角速度

generator = Generator(z_dim, hidden_dim, output_dim)
discriminator = Discriminator(output_dim, hidden_dim)

generator.load_state_dict(torch.load('../model/generator.pth', map_location=device))
discriminator.load_state_dict(torch.load('../model/discriminator.pth', map_location=device))


z = torch.randn(1, slice_length, z_dim)
generated_sequence = generator(z)

data = generated_sequence[0].detach().numpy()

x = range(data.shape[0])


dim1, dim2, dim3 = data[:, 0], data[:, 1], data[:, 2]

# 绘制前3维的时间序列
plt.figure(figsize=(12, 6))
plt.plot(x, dim1, label='Dimension 1')
plt.plot(x, dim2, label='Dimension 2')
plt.plot(x, dim3, label='Dimension 3')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Visualization of the First 3 Dimensions of 9D Data')
plt.legend()
plt.grid(True)
plt.show()