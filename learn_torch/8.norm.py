import torch

# 创建一个张量
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 计算L2范数（默认p=2）
norm_2 = torch.norm(x)

# 计算L1范数
norm_1 = torch.norm(x, p=1)

# 计算无穷范数
norm_inf = torch.norm(x, p=float('inf'))

print(norm_2)   # 输出: tensor(5.4772)
print(norm_1)   # 输出: tensor(10.)
print(norm_inf) # 输出: tensor(4.)
