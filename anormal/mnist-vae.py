import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 定义 VAE 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar


# 重新参数化技巧：将均值和对数方差转换为潜在空间向量
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# 定义 VAE 解码器
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))


# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


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
vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)


# 训练 VAE 模型
def train_vae():
    vae.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
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
        recon_data, _, _ = vae(test_data)

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
