import numpy as np
import torch
from matplotlib import pyplot as plt

from datareader.mh_datareader import simple_get_mh_all_features
from datareader.realworld_datareader import simple_get_realworld_all_features
from gan.lstm_gan import Generator, Discriminator
from prototype import constant

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

slice_length = 256
z_dim = 9       # 噪声向量的维度
hidden_dim = 30  # LSTM 的隐藏层维度
output_dim = 9   # 时间序列的特征维度，即加速度和角速度
slice_length = 256
filtered_label = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
mapping = constant.Constant.simple_action_set.mapping_mh
###############################################################
origin_data = simple_get_mh_all_features(slice_length, type='np',
                                         filtered_label=filtered_label,
                                         mapping_label=mapping, with_rpy=False, need_transform=False)

# 去除标签
origin_data = origin_data[:,:,:9].astype(np.float32)
###############################################################
filtered_label_real_world = [0, 1, 2, 3, 4, 5, 6]
mapping_realworld = constant.Constant.simple_action_set.mapping_realworld
target_data = simple_get_realworld_all_features(slice_length, type='np',
                                                filtered_label=filtered_label_real_world,
                                                mapping_label=mapping_realworld,
                                                with_rpy=False, need_transform=False)

target_data = target_data[:, :, :9].astype(np.float32)
###############################################################
G = Generator(z_dim, hidden_dim, output_dim)
F = Generator(z_dim, hidden_dim, output_dim)
D_X = Discriminator(output_dim, hidden_dim).to(device)
D_Y = Discriminator(output_dim, hidden_dim).to(device)

G.load_state_dict(torch.load('../model/cycle_g_generator.pth', map_location=device, weights_only=True))
F.load_state_dict(torch.load('../model/cycle_f_generator.pth', map_location=device, weights_only=True))

D_X.load_state_dict(torch.load('../model/cycle_dx_discriminator.pth', map_location=device, weights_only=True))
D_Y.load_state_dict(torch.load('../model/cycle_dy_discriminator.pth', map_location=device, weights_only=True))



# 原始数据
random_sample = origin_data[np.random.choice(origin_data.shape[0])][np.newaxis, ...]
random_sample2 = target_data[np.random.choice(target_data.shape[0])]

generated_sequence = G(torch.tensor(random_sample))
random_sample = random_sample[0]
data = generated_sequence[0].detach().numpy()

x = range(data.shape[0])

# 生成数据
dim1, dim2, dim3 = data[:, 0], data[:, 1], data[:, 2]


plt.figure(figsize=(30, 8))
# 真实数据
plt.subplot(3, 1, 1)
plt.plot(x, random_sample[:,0], label='origin acc_x', color='red')
plt.plot(x, random_sample[:,1], label='origin acc_y', color='green')
plt.plot(x, random_sample[:,2], label='origin acc_z', color='blue')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Real Data')
plt.grid(True)
plt.legend()


# 真实数据2
plt.subplot(3, 1, 2)
plt.plot(x, random_sample2[:,0], label='target acc_x', color='red')
plt.plot(x, random_sample2[:,1], label='target acc_y', color='green')
plt.plot(x, random_sample2[:,2], label='target acc_z', color='blue')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Target Domain Data')
plt.grid(True)
plt.legend()


# 虚假数据
plt.subplot(3, 1, 3)
plt.plot(x, dim1, label='generated acc_x', color='red')
plt.plot(x, dim2, label='generated acc_y', color='green')
plt.plot(x, dim3, label='generated acc_z', color='blue')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Fake Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()