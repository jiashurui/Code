import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from anormal.AEModel import VAE, ConvAutoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据准备
# transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# For CAE
# transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(1, -1))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 参数设置
input_dim = 28 * 28  # MNIST 图像尺寸为28x28
hidden_dim = 400
latent_dim = 20
learning_rate = 1e-3
num_epochs = 10

# 初始化模型和优化器
# model = VAE(input_dim=input_dim, z_dim=20)


# CAE
# train_data = train_normal.transpose(1, 2)
# test_data = train_abnormal.transpose(1, 2)
# input_dim = train_data.size(2)  # dim for CNN is changed
loss_function = nn.MSELoss()  # MSE loss for reconstruction

model = ConvAutoencoder(input_dim = 1).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print('start train')
# 训练 VAE 模型
def train_vae():
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()


            # recon_batch,_, mu, logvar = model(data)
            # loss = model.loss_function(recon_batch, data, mu, logvar)

            output = model(data)
            loss = loss_function(output, data)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')


# 训练模型
train_vae()


# 重构测试数据并可视化
def visualize_reconstruction():
    model.eval()
    with torch.no_grad():
        # 获取一个batch的测试数据
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
        test_data, _ = next(iter(test_loader))

        # 重构图像
        # recon_data,_, _, _ = model(test_data)

        recon_data = model(test_data)

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
