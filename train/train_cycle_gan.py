import numpy as np
import torch
from torch.utils.data import DataLoader

from datareader.mh_datareader import simple_get_mh_all_features
from datareader.realworld_datareader import simple_get_realworld_all_features
from gan.cycle_gan import Generator, Discriminator, adversarial_loss, cycle_consistency_loss
from prototype import constant
from utils.pair_dataloader import PairedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 1000
batch_size = 64

slice_length = 256



################################################
#  Origin Data
filtered_label = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
mapping = constant.Constant.simple_action_set.mapping_mh
origin_data = simple_get_mh_all_features(slice_length, type='np',
                                         filtered_label=filtered_label,
                                         mapping_label=mapping, with_rpy=False)
# 去除标签
origin_data = origin_data[:,:,:9].astype(np.float32)
# data_loader = DataLoader(origin_data, batch_size=batch_size, shuffle=True)
################################################
#  Target Data

filtered_label_real_world = [0, 1, 2, 3, 5]
mapping_realworld = constant.Constant.simple_action_set.mapping_realworld

# 全局变换之后RealWorld数据(全局变换按照frame进行)
target_data = simple_get_realworld_all_features(slice_length, type='df',
                                                filtered_label=filtered_label_real_world,
                                                mapping_label=mapping_realworld,
                                                with_rpy=False)

paired_dataset = PairedDataset(origin_data, target_data)
data_loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)


# Network Param
z_dim = 10       # 噪声向量的维度
hidden_dim = 30  # LSTM 的隐藏层维度
output_dim = 9   # 时间序列的特征维度，即加速度和角速度


# 定义模型
G = Generator(z_dim, hidden_dim, output_dim).to(device)
F = Generator(z_dim, hidden_dim, output_dim).to(device)
D_X = Discriminator(output_dim, hidden_dim).to(device)
D_Y = Discriminator(output_dim, hidden_dim).to(device)

# 优化器
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
f_optimizer = torch.optim.Adam(F.parameters(), lr=0.0002, betas=(0.5, 0.999))
dx_optimizer = torch.optim.Adam(D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
dy_optimizer = torch.optim.Adam(D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练循环
for epoch in range(epochs):
    for real_x, real_y in data_loader:  # real_x 来自源域，real_y 来自目标域
        real_x, real_y = real_x.to(device), real_y.to(device)

        # 标签定义
        real_label = torch.ones(real_x.size(0), 1).to(device)
        fake_label = torch.zeros(real_x.size(0), 1).to(device)

        # --------------------
        # 训练判别器 D_X
        # --------------------
        fake_x = F(real_y)
        dx_real_loss = adversarial_loss(D_X(real_x), real_label)
        dx_fake_loss = adversarial_loss(D_X(fake_x.detach()), fake_label)
        dx_loss = (dx_real_loss + dx_fake_loss) / 2
        dx_optimizer.zero_grad()
        dx_loss.backward()
        dx_optimizer.step()

        # --------------------
        # 训练判别器 D_Y
        # --------------------
        fake_y = G(real_x)
        dy_real_loss = adversarial_loss(D_Y(real_y), real_label)
        dy_fake_loss = adversarial_loss(D_Y(fake_y.detach()), fake_label)
        dy_loss = (dy_real_loss + dy_fake_loss) / 2
        dy_optimizer.zero_grad()
        dy_loss.backward()
        dy_optimizer.step()

        # --------------------
        # 训练生成器 G 和 F
        # --------------------
        # GAN Loss
        g_loss = adversarial_loss(D_Y(fake_y), real_label)
        f_loss = adversarial_loss(D_X(fake_x), real_label)

        # Cycle Consistency Loss
        cycle_x = F(fake_y)
        cycle_y = G(fake_x)
        cycle_loss_x = cycle_consistency_loss(real_x, cycle_x)
        cycle_loss_y = cycle_consistency_loss(real_y, cycle_y)
        cycle_loss = cycle_loss_x + cycle_loss_y

        # 总生成器损失
        total_g_loss = g_loss + f_loss + 10 * cycle_loss
        g_optimizer.zero_grad()
        f_optimizer.zero_grad()
        total_g_loss.backward()
        g_optimizer.step()
        f_optimizer.step()

    print(f"Epoch [{epoch}/{epochs}], D_X Loss: {dx_loss.item()}, D_Y Loss: {dy_loss.item()}, G Loss: {total_g_loss.item()}")

# 假设 generator 和 discriminator 是训练好的生成器和判别器模型
# 保存生成器的参数
torch.save(G.state_dict(), "../model/cycle_g_generator.pth")
torch.save(F.state_dict(), "../model/cycle_f_generator.pth")

# 保存判别器的参数
torch.save(D_X.state_dict(), "../model/cycle_dx_discriminator.pth")
torch.save(D_Y.state_dict(), "../model/cycle_dy_discriminator.pth")
