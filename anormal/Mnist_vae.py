import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from anormal.AEModel import VAE

# 数据准备
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 参数设置
input_dim = 28 * 28  # MNIST 图像尺寸为28x28
hidden_dim = 400
latent_dim = 20
learning_rate = 1e-3
num_epochs = 10

# 初始化模型和优化器
vae = VAE(input_dim=input_dim, z_dim=20)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)


# 训练 VAE 模型
def train_vae():
    vae.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch,_, mu, logvar = vae(data)
            loss = vae.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')


# 训练模型
train_vae()


# 重构测试数据并可视化
def visualize_reconstruction():
    vae.eval()
    with torch.no_grad():
        # 获取一个batch的测试数据
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
        test_data, _ = next(iter(test_loader))

        # 重构图像
        recon_data,_, _, _ = vae(test_data)

        # 可视化原始图像和重构图像
        test_data = test_data.view(-1, 28, 28)
        recon_data = recon_data.view(-1, 28, 28)

        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        for i in range(10):
            axes[0, i].imshow(test_data[i], cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon_data[i], cmap='gray')
            axes[1, i].axis('off')
        plt.show()


# 可视化重构结果
visualize_reconstruction()
