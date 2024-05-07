import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

# dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义第一层（输入层到隐藏层）
        self.layer1 = nn.Linear(28 * 28 , 128)
        # 定义第二层（隐藏层到输出层）
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        # flatten
        x = x.view(-1, 28*28)
        # 输入数据通过第一层，然后应用ReLU激活函数
        x = F.relu(self.layer1(x))
        # 通过第二层得到输出
        x = self.layer2(x)
        # log softmax is better
        x = F.log_softmax(x, dim=1)
        return x
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失加起来
            pred = output.argmax(dim=1, keepdim=True)  # 获取概率最大的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 定义数据转换，转为tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

# 下载训练集
train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 下载测试集
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 创建网络实例
model = Model()

# 损失函数
loss = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1, 10):  # 训练10轮
    train(model, train_loader, optimizer, loss, epoch)

# 保存模型
# torch.save(model, 'model.pth')

# 保存模型参数（推荐方式）
torch.save(model.state_dict(), '../model/MNIST_NN.pth')

# 加载整个模型
# model_load = torch.load('../model/MNIST_NN.pth')

# 实例化模型(加载模型参数)
model_load = Model()
model_load.load_state_dict(torch.load('../model/MNIST_NN.pth'))

test(model_load, test_loader, loss)



