# 设置训练参数
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datareader.mh_datareader import simple_get_mh_all_features
from datareader.realworld_datareader import simple_get_realworld_all_features
from gan.lstm_gan import Generator, Discriminator
from prototype import constant
from utils import show

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 1000
batch_size = 64

slice_length = 256
filtered_label = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
mapping = constant.Constant.simple_action_set.mapping_mh

# 全局变换之后的大学生数据(全局变换按照frame进行)
origin_data = simple_get_mh_all_features(slice_length, type='np',
                                         filtered_label=filtered_label,
                                         mapping_label=mapping, with_rpy=False)
# 去除标签
origin_data = origin_data[:,:,:9].astype(np.float32)
data_loader = DataLoader(origin_data, batch_size=batch_size, shuffle=True)


z_dim = 10       # 噪声向量的维度
hidden_dim = 30  # LSTM 的隐藏层维度
output_dim = 9   # 时间序列的特征维度，即加速度和角速度

generator = Generator(z_dim, hidden_dim, output_dim).to(device)
discriminator = Discriminator(output_dim, hidden_dim).to(device)

# 优化器
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

g_loss_arr = []
d_loss_arr = []

# 训练循环
for epoch in range(epochs):

    g_loss_per_epoch = 0.0
    d_loss_per_epoch = 0.0

    for real_data in data_loader:  # data_loader 应该提供 shape 为 (batch_size, seq_len, output_dim) 的真实数据
        real_data = real_data.to(device)  # real_data 是一个元组，取出实际的张量

        batch_size = real_data.size(0)

        # 生成噪声
        z = torch.randn(batch_size, slice_length, z_dim).to(device)

        # 生成假数据
        fake_data = generator(z)

        # 判别器训练：分辨真实和生成数据
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # 判别器损失（真实数据）
        real_preds = discriminator(real_data)
        d_loss_real = nn.BCELoss()(real_preds, real_labels)

        # 判别器损失（生成数据）
        fake_preds = discriminator(fake_data.detach())
        d_loss_fake = nn.BCELoss()(fake_preds, fake_labels)

        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 生成器训练：希望生成的数据被判别器认为是真实的
        fake_preds = discriminator(fake_data)
        g_loss = nn.BCELoss()(fake_preds, real_labels)  # 生成器的目标是让判别器认为生成数据是真实的

        # 累加每个batch_size损失
        g_loss_per_epoch = g_loss_per_epoch + g_loss.item()/batch_size
        d_loss_per_epoch = d_loss_per_epoch + d_loss.item()/batch_size

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    # 记录数据
    g_loss_arr.append(g_loss_per_epoch)
    d_loss_arr.append(d_loss_per_epoch)


    print(f"Epoch [{epoch}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

show.show_me_data0(g_loss_arr)
show.show_me_data0(d_loss_arr)

# 假设 generator 和 discriminator 是训练好的生成器和判别器模型
# 保存生成器的参数
torch.save(generator.state_dict(), "../model/generator.pth")

# 保存判别器的参数
torch.save(discriminator.state_dict(), "../model/discriminator.pth")
