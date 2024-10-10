import torch

# 假设我们有一个形状为 (N, *) 的张量
tensor = torch.randn(10, 3)  # 示例张量，形状为 (10, 3)

# 获取第一个维度的大小
N = tensor.size(0)

# 生成随机采样的索引
random_indices = torch.randperm(5)

# 对第一个维度进行采样
sampled_tensor = tensor[random_indices]

print(sampled_tensor)
