import torch

from gan.lstm_gan import Generator, Discriminator

batch_size = 64

slice_length = 256
z_dim = 10       # 噪声向量的维度
hidden_dim = 30  # LSTM 的隐藏层维度
output_dim = 9   # 时间序列的特征维度，即加速度和角速度

generator = Generator(z_dim, hidden_dim, output_dim)
discriminator = Discriminator(output_dim, hidden_dim)

generator.load_state_dict(torch.load('../model/generator.pth', map_location=torch.device('cpu')))
discriminator.load_state_dict(torch.load('../model/discriminator.pth'))


z = torch.randn(batch_size, slice_length, z_dim)
generated_sequence = generator(z)
